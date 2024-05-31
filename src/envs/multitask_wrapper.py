from gymnasium import spaces
from gymnasium.wrappers import TransformObservation
import gymnasium as gym
import numpy as np 

from PyFlyt.gym_envs import FlattenWaypointEnv
from envs.obstacle_hover_wrapper import QuadXHoverExtendedObservation
from envs.obstacle_env import QuadXObstacleEnv

class MultitaskWrapper(gym.Env):

    def __init__(self, all=False) -> None:
        super().__init__()

        angular_rate_limit = np.pi
        thrust_limit = 0.8
        high = np.array(
            [
                angular_rate_limit,
                angular_rate_limit,
                angular_rate_limit,
                thrust_limit,
            ]
        )
        low = np.array(
            [
                -angular_rate_limit,
                -angular_rate_limit,
                -angular_rate_limit,
                0.0,
            ]
        )

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,)
        )

        waypoint_env = TransformObservation(FlattenWaypointEnv(gym.make("PyFlyt/QuadX-Waypoints-v1"), context_length=1) , lambda obs: np.append( obs,[-5, -5, 5]))
        if all:
            self.envs = [QuadXHoverExtendedObservation(),waypoint_env, QuadXObstacleEnv()] # QuadXHoverExtendedObservation
        else:
            self.envs = [waypoint_env, QuadXObstacleEnv()]
        self.env_index = 0

    def next_env(self):
        self.env_index = np.random.choice(len(self.envs ))
        #self.env_index = (self.env_index + 1) % len(self.envs )
        self.reset()

    def set_env_index(self, index):
        self.env_index = index
        self.reset()

    def step(self, action):
        return self.envs[self.env_index].step(action)

    def reset(self, seed=None, options=None):
        return self.envs[self.env_index].reset()

    def render(self):
        self.envs[self.env_index].render()

    def close(self):
        for env in self.envs:
            env.close()