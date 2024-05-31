import random
import datetime
import os

import torch 
import torch.optim as optim

import numpy as np
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

from option_critic import OptionCritic
from rollout_buffer import RolloutBuffer
from envs.multitask_wrapper import MultitaskWrapper
from envs.obstacle_env import QuadXObstacleEnv
from envs.obstacle_hover_wrapper import QuadXHoverExtendedObservation

# Partially based on https://github.com/ShangtongZhang/DeepRL
def tensor(x, device="cuda"):
    if torch.is_tensor(x):
        return x
    if not torch.cuda.is_available():
            device = "cpu"
    x = np.asarray(x, dtype=np.float32)
    x = torch.tensor(x, device=torch.device(device), dtype=torch.float32)
    return x

def make_env(environment_id, log_dir):
    def _thunk():
        info_keywords = ()
        if environment_id=="QuadX-Hover-v1":
            env = gym.make(f"PyFlyt/{environment_id}")
        elif environment_id=="QuadX-Hover-Extended-v1":
            env = QuadXHoverExtendedObservation()
        elif environment_id=="QuadX-Waypoints-v1":
            env = gym.make(f"PyFlyt/{environment_id}")
            info_keywords = ("num_targets_reached",)
            env = FlattenWaypointEnv(env, context_length=1)   
        elif environment_id=="QuadX-Obstacles-v1":
            info_keywords = ("num_targets_reached",)
            env = QuadXObstacleEnv()
        elif environment_id=="transferlearning":
            info_keywords = ("num_targets_reached",)
            env= EnvWrapper()
        elif environment_id=="transferlearning-all":
            env = EnvWrapper(all=all)
        else:
            raise "Uncompatible environment"
        env = gym.wrappers.NormalizeObservation(env)
        env = Monitor(env, log_dir, info_keywords=info_keywords)
        return env
    return _thunk


class PPOC():
    def __init__(self,
                 num_envs=16,
                 env_name="HalfCheetah-v2",
                 lr=3e-4,
                 rollout_steps=2048,
                 num_options=4,
                 ppo_epochs=10,
                 mini_batch_size=64,
                 device="cuda"):
        self.num_envs = num_envs
        self.env_name = env_name
        self.curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S") \
                    + "_{:04d}".format(random.randint(1,9999))
        self.path='./data2/{}'.format(self.env_name)
        try:
            os.mkdir(self.path)
        except OSError as error:
            print(error)
        self.log_dir = self.path + "/" + self.curtime

        self.rollout_steps = rollout_steps
        self.lr = lr
        self.envs = make_vec_env(make_env(env_name, self.log_dir), n_envs=self.num_envs)
        self.num_inputs = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.shape[0]
        self.num_options = num_options
        self.is_init_state = tensor(np.ones((self.num_envs)), device=device).byte()
        self.worker_index = tensor(np.arange(self.num_envs), device=device).long()
        self.prev_option = tensor(np.zeros(self.num_envs), device=device).long()

        self.model = OptionCritic(self.num_inputs, self.num_outputs, self.num_options, device=device, worker_index=self.worker_index)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr,
                                    eps=1e-5)
        
        self.eps = 0.1
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.rollout_buffer = RolloutBuffer(
            action_dim=self.num_outputs,
            obs_shape=self.num_inputs,
            buffer_size=self.rollout_steps,
            n_envs=self.num_envs,
            n_options=self.num_options,
            batch_size=self.mini_batch_size,
            device=self.model.device
        )


    def ppo_update(self,
               clip_param=0.2):
        for _ in range(self.ppo_epochs):
            for rollout_data in self.rollout_buffer:
                option_probs = self.model.evaluate_options(rollout_data.observations)
                values, new_log_probs, entropy = self.model.evaluate_actions(
                    rollout_data.observations, 
                    rollout_data.options, 
                    rollout_data.actions)

                ratio = torch.exp(new_log_probs - rollout_data.old_log_prob)
                surr1 = ratio * rollout_data.advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                    1.0 + clip_param) * rollout_data.advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (rollout_data.returns - values.gather(1, rollout_data.options)).pow(2).mean()
                beta_loss = (rollout_data.betas.gather(1, rollout_data.prev_options) * rollout_data.beta_advantages * (1 - rollout_data.episode_starts)).mean()
        
                policy_over_option_loss = -(option_probs.gather(1, rollout_data.options) * rollout_data.advantages).mean() - 0.01 * (-(option_probs*option_probs.log()).sum(-1).mean())#entropy
                loss = 0.5 * critic_loss + actor_loss  + beta_loss + policy_over_option_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
    
    def callback_end_episode(self, done):
        for i in range(self.num_envs):
            if done[i]:             
                print("Cumulative reward at step " + str(self.episode) +
                        " is " + str(self.cumu_rewd[i]))
                self.fd_train.write("%d %f\n" % (self.episode, self.cumu_rewd[i]))
                self.episode +=1
                if self.episode == 4000 and self.env_name=="QuadX-Hover-Extended-v1":
                    self.envs.env_method("change_start")
                    self.envs.reset()
                if self.env_name=="transferlearning-all":
                            self.envs.env_method("next_env")
                if self.episode == 4000 and self.env_name=="QuadX-Hover-Extended-v1":
                    self.envs.env_method("change_start")
                    self.envs.reset()
                if self.episode == 4000 and self.env_name=="transferlearning":
                    self.envs.env_method("set_env_index",1)
                    self.envs.reset()
                self.cumu_rewd[i] = 0

            self.fd_train.flush()


    def collect_rollouts(self):
        n_steps = 0
        masks = []
        values = []
        while n_steps < self.rollout_steps:
            state = torch.FloatTensor(self.last_state).to(self.model.device)
            with torch.no_grad():
                action, beta, option= self.model(state, self.prev_option, self.is_init_state)
                q_option, log_prob, entropy = self.model.evaluate_actions(
                        state, 
                        option, 
                        action)
            
            next_state, reward, done, info = self.envs.step(action.cpu().detach().numpy())
            self.cumu_rewd += reward
            
            self.callback_end_episode(done)

            self.rollout_buffer.store(
                observations=state,
                actions=action,
                betas=beta,
                option_values=q_option,
                log_probs=log_prob,
                entropies=entropy,
                options=option.unsqueeze(-1),
                prev_options=self.prev_option.unsqueeze(-1),
                episode_starts=self.is_init_state.unsqueeze(-1).float(),
                rewards=torch.FloatTensor(reward).unsqueeze(-1).to(self.model.device)
            )

            masks.append(torch.FloatTensor(1 - done).unsqueeze(-1).to(self.model.device))
            values.append(q_option[self.worker_index, option].unsqueeze(-1))
            self.is_init_state = tensor(done, device=self.model.device).byte()
            self.prev_option = option
            self.last_state = next_state
            self.current_timestep += self.num_envs
            n_steps +=1

        next_state = torch.FloatTensor(self.last_state).to(self.model.device)
        with torch.no_grad():
            action, beta, option= self.model(state, self.prev_option, self.is_init_state)
            option_probs = self.model.evaluate_options(state)
            q_option, log_prob, entropy = self.model.evaluate_actions(
                    state, 
                    option, 
                    action)

        self.rollout_buffer.calculate_advantages(
            self.prev_option, 
            masks, 
            values, 
            beta, 
            q_option, 
            option_probs, 
            self.worker_index)

    def learn(self, num_timesteps=2000000):
        self.episode = 1
        self.current_timestep = 0     
        self.cumu_rewd = np.zeros(self.num_envs)
        
        self.fd_train = open(self.path + '/ppoc_train_{}.log'.format(self.curtime), 'w')
        while self.current_timestep < num_timesteps:
            self.rollout_buffer.reset()
            self.last_state = self.envs.reset()
            self.collect_rollouts()
            self.ppo_update()
        self.fd_train.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run PPOC on a specific game.')
    parser.add_argument('-e', '--env_name', type=str, help='Name of the game', default="QuadX-Hover-v1") #"QuadX-Hover-v1"
    parser.add_argument('-n', '--num_envs', type=int, help='Number of workers', default=1)
    args = parser.parse_args()
    ocagent = PPOC(num_envs=args.num_envs, env_name=args.env_name, num_options=2)
    ocagent.learn()