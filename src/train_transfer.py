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

from envs.multitask_wrapper import MultitaskWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

def make_env(log_dir, all=False, norm_reward=False):
    def _thunk():
        env = MultitaskWrapper(all=all)
        env = gym.wrappers.NormalizeObservation(env)
        if norm_reward:
            env = gym.wrappers.NormalizeReward(env,gamma=0.99, epsilon=1e-08)
        if not all:
            info_keywords = ("num_targets_reached",)
            env = Monitor(env, log_dir,info_keywords=info_keywords)
        env = Monitor(env, log_dir)
        return env
    return _thunk

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
        self.was_changed = False
    
    def _on_step(self) -> bool:
        self.n_episodes += np.sum(self.locals["dones"]).item()
        if self.n_episodes == 4000 and not self.was_changed:
        #if self.n_episodes > 200000 and self.n_episodes < 200000 + self.training_env.num_envs:
            self.training_env.env_method("set_env_index",1)
            self.was_changed = True
        return True
    
class ChangeEnvsCallback(BaseCallback):
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
        if self.n_episodes % 5000==0 and np.sum(self.locals["dones"]).item() > 0:
            train_env = self.training_env
            train_env.env_method("set_env_index", 0)
            mean_reward, std_reward = evaluate_policy(self.model, train_env, n_eval_episodes=100)
            print(f"Hover: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
            train_env.env_method("set_env_index", 1)
            mean_reward, std_reward = evaluate_policy(self.model, train_env, n_eval_episodes=100)
            print(f"Waypoint: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
            train_env.env_method("set_env_index", 2)
            mean_reward, std_reward = evaluate_policy(self.model, train_env, n_eval_episodes=100)
            print(f"Obstacles: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        
        if np.sum(self.locals["dones"]).item() >0:
            self.training_env.env_method("next_env")
        
        return True

def train(args):
    environment = args['environment'] 
    id = ''.join(random.choices(string.ascii_letters, k=20))
    if environment == "Transferlearning_Waypoint_Obstacle":
        full_id = "PPO2_" + '_Transferlearning_' + id
    else:
        full_id = "PPO2_" + '_Transferlearning_all_' + id

    models_dir = f"models/{full_id}/"
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
            log_dir=logdir if args["log"]else None, all= environment != "Transferlearning_Waypoint_Obstacle"), 
        n_envs=1,)
        #vec_env_cls=DummyVecEnv)

    change_start_callback = ChangeStartCallback()
    iterate_envs_callback = ChangeEnvsCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000,
        save_path=checkpoint_dir,
        name_prefix="PPO_" + environment
    )
    if environment == "Transferlearning_Waypoint_Obstacle":
        callbacks = CallbackList([checkpoint_callback, change_start_callback])
    else:
        callbacks = CallbackList([checkpoint_callback, iterate_envs_callback])

    model = PPO(
            policy="MlpPolicy", 
            env=train_env, 
            verbose=1, 
            tensorboard_log=logdir if args["log"] else None)

    try:   
        model.learn(total_timesteps=args["total_timesteps"],
                    callback=callbacks,
                    tb_log_name="tensorboard")
    
    except KeyboardInterrupt:
        model_name = f"{checkpoint_dir}/{model.num_timesteps}"
        model.save(model_name)
        with open('recent_model.txt', 'w') as file:
            file.write(model_name)
    if environment != "Transferlearning_Waypoint_Obstacle":
        train_env.env_method("set_env_index", 0)
        mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
        print(f"Hover: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        train_env.env_method("set_env_index", 1)
        mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
        print(f"Waypoint: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        train_env.env_method("set_env_index", 2)
        mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
        print(f"Obstacles: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    train_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in an environment using a DRL algorithm from stable baselines 3')

    parser.add_argument('--log', type=bool, default=True, help='record logs to tensorboard')
    parser.add_argument('--total_timesteps', '-tim', type=int, default=4000000, help='training duration')
    parser.add_argument('--environment', '-env', type=str, default="Transferlearning_Waypoint_Obstacle", help='which environment to train on')#"Transferlearning_Waypoint_Obstacle"

    args = parser.parse_args()
    
    train(vars(args))