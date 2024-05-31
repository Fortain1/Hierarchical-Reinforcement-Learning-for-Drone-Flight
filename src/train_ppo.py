import gymnasium as gym
import os
import random
import string
import argparse
import numpy as np
import torch

from envs.obstacle_hover_wrapper import QuadXHoverExtendedObservation
from envs.obstacle_env import QuadXObstacleEnv
from PyFlyt.gym_envs import FlattenWaypointEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

def make_env(environment_id, log_dir):
    def _thunk():
        info_keywords = ()
        if environment_id=="QuadX-Hover-v1":
            env = gym.make(f"PyFlyt/{environment_id}")
        elif environment_id=="QuadX-Hover-Extended-v1":
            env = QuadXHoverExtendedObservation()
        elif environment_id=="QuadX-Waypoints-v1":
            info_keywords = ("num_targets_reached",)
            env = gym.make(f"PyFlyt/{environment_id}")
            env = FlattenWaypointEnv(env, context_length=1)   
        elif environment_id=="QuadX-Obstacles-v1":
            info_keywords = ("num_targets_reached",)
            env = QuadXObstacleEnv()
        else:
            raise "Uncompatible environment"
        env = gym.wrappers.NormalizeObservation(env)
        env = Monitor(env, log_dir, info_keywords=info_keywords)
        return env
    return _thunk

def evaluate_policy(
    model,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    waypoints = False
):
    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    waypoints_reached = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if dones[i]:         
                    if "episode" in info.keys():
                        # Do not trust "done" with episode endings.
                        # Monitor wrapper includes "episode" key in info if environment
                        # has been wrapped with it. Use those rewards instead.
                        episode_rewards.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        if "num_targets_reached" in info.keys():
                            waypoints_reached.append(info["num_targets_reached"])
                        # Only increment at the real end of an episode
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if waypoints:
        return mean_reward, std_reward, np.mean(waypoints_reached), np.std(waypoints_reached)
    return mean_reward, std_reward

class ChangeStartCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.n_episodes = 0
    

    def _on_step(self) -> bool:
        self.n_episodes += np.sum(self.locals["dones"]).item()
        if self.n_episodes == 3000:
        #if self.n_episodes > 200000 and self.n_episodes < 200000 + self.training_env.num_envs:
            self.training_env.env_method("change_start", goal= np.array([[-1.0, -1.0, 1.0]]))
        return True

def train(args):
    experiment = args['environment'] + "_2"
    environment = args['environment']
    id = ''.join(random.choices(string.ascii_letters, k=20))
    full_id = "PPO_" + '_' + environment + '_' + id

    models_dir = f"models/{experiment}/{full_id}/"
    logdir = models_dir + "data"
    checkpoint_dir = models_dir + "checkpoints/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    
    train_env = make_vec_env(
        make_env(
            environment_id=environment, 
            log_dir=logdir if args["log"]else None), 
        n_envs=1,)
        #vec_env_cls=DummyVecEnv)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    change_start_callback = ChangeStartCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=1000000,
        save_path=checkpoint_dir,
        name_prefix="PPO_" + environment
    )

    callbacks = CallbackList([checkpoint_callback, change_start_callback])
    if args["sde"]:
        policy_kwargs = dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256,256], vf=[256,256])
        )
        model = PPO(
            policy="MlpPolicy", 
            env=train_env,
            batch_size=128,
            n_steps=512,
            gamma=0.99,
            n_epochs=40,
            gae_lambda=0.9,
            ent_coef=0.0,
            sde_sample_freq=4,
            max_grad_norm=0.5,
            vf_coef=0.5,
            learning_rate=3e-5,
            use_sde=True,
            clip_range=0.4,
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            tensorboard_log=logdir if args["log"] else None)
    else:
        model = PPO(
            policy="MlpPolicy", 
            env=train_env, 
            verbose=1, 
            tensorboard_log=logdir if args["log"] else None)
        train_env.save(checkpoint_dir+"env.pkl")

    try:
        if environment == "QuadX-Hover-Extended-v1":
            model.learn(total_timesteps=args["total_timesteps"],
                    callback=callbacks,
                    tb_log_name="tensorboard")
        else:
            model.learn(total_timesteps=args["total_timesteps"],
                    callback=checkpoint_callback,
                    tb_log_name="tensorboard")
    except KeyboardInterrupt:
        model_name = f"{checkpoint_dir}/{model.num_timesteps}"
        model.save(model_name)
        with open('recent_model.txt', 'w') as file:
            file.write(model_name)
    if environment == "QuadX-Waypoints-v1" or environment == "QuadX-Obstacles-v1":
        mean_reward, std_reward, waypoint_mean, waypoint_std = evaluate_policy(model, train_env, n_eval_episodes=100, waypoints=True)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"mean_reached_waypoints:{waypoint_mean:.2f} +/- {waypoint_std:.2f}")
    else: 
        mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
     
    train_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in an environment using a DRL algorithm from stable baselines 3')

    parser.add_argument('--sde', type=bool, default=False, help='uses sde')
    parser.add_argument('--log', type=bool, default=True, help='record logs to tensorboard')
    parser.add_argument('--environment', '-env', type=str, default="QuadX-Obstacles-v1", help='which environment to train on')#QuadX-Hover-Extended-v1
    parser.add_argument('--total_timesteps', '-tim', type=int, default=1000000, help='training duration')
    args = parser.parse_args()
    train(vars(args))