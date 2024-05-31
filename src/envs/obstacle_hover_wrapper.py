"""QuadX Waypoints Environment."""
from __future__ import annotations

from typing import Literal
from gymnasium import spaces

import numpy as np
from PyFlyt.gym_envs.quadx_envs.quadx_hover_env import QuadXHoverEnv, QuadXBaseEnv


class QuadXHoverExtendedObservation(QuadXBaseEnv):

    """QuadX Waypoints Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is a set of `[x, y, z, (optional) yaw]` waypoints in space.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        num_targets (int): number of waypoints in the environment.
        use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
        goal_reach_distance (float): distance to the waypoints for it to be considered reached.
        goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
        flight_mode (int): the flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (str): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | str): can be "human" or None.
        render_resolution (tuple[int, int]): render_resolution.
    """
    def __init__(
        self,
        sparse_reward: bool = False,
        flight_dome_size: float = 3.0,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
        goal_pos = np.array([1.0, 1.0, 1.0]),
        start_pos = np.array([[0, 0, 1]])
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (str): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | str): can be "human" or None.
            render_resolution (tuple[int, int]): render_resolution.
        """
        super().__init__(
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
            start_pos=start_pos,
        )

        """GYMNASIUM STUFF"""
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.combined_space.shape[0] + 6,)
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward

        self.goal_pos = goal_pos

    def change_start(self, goal= np.array([[-1.0, -1.0, 1.0]])):
        self.goal = goal

    def reset(self, seed=None, options=dict()):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """
        super().begin_reset(seed, options)
        super().end_reset(seed, options)
        return self.state, self.info

    def compute_state(self):
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 4 values)
        - auxiliary information (vector of 4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        rotation = np.array(self.env.getMatrixFromQuaternion(quarternion)).reshape(3, 3)
        distance_to_target =  np.matmul((self.goal_pos - lin_pos), rotation)
        # combine everything
        if self.angle_representation == 0:
            self.state = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action, *aux_state, *distance_to_target, -100, -100, -100]
            )
        elif self.angle_representation == 1:
            self.state = np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action, *aux_state, *distance_to_target, -100, -100, -100]
            )

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        if not self.sparse_reward:
            # distance from 0, 0, 1 hover point
            linear_distance = np.linalg.norm(
                self.env.state(0)[-1] - self.goal_pos
            )
            if linear_distance < 0.3:
                # how far are we from 0 roll pitch
                angular_distance = np.linalg.norm(self.env.state(0)[1][:2])
                self.reward -= linear_distance + angular_distance
            else:
                self.reward -= linear_distance

            
            self.reward += 1.0
    



