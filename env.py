import gymnasium as gym
from gymnasium import spaces
import numpy as np
from physics import step_ball

class GolfPuttingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.hole_radius = 0.04

        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0]),
            high=np.array([np.pi, 1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        self.reset()

    def set_positions(self, ball, hole):
        self.ball_pos = np.array(ball, dtype=np.float32)
        self.hole_pos = np.array(hole, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.has_struck = False
        self.steps = 0

    def reset(self, seed=None, options=None):
        self.ball_pos = np.array([0.2, 0.5], dtype=np.float32)
        self.hole_pos = np.array([0.8, 0.5], dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.has_struck = False
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.ball_pos,
            self.ball_vel,
            self.hole_pos
        ])

    def step(self, action):
        reward = 0
        self.steps += 1

        if not self.has_struck:
            angle, power = action
            self.ball_vel = np.array([
                np.cos(angle),
                np.sin(angle)
            ]) * power * 0.25
            self.has_struck = True

        self.ball_pos, self.ball_vel = step_ball(
            self.ball_pos, self.ball_vel
        )

        dist = np.linalg.norm(self.ball_pos - self.hole_pos)
        reward -= dist

        done = False
        success = False

        if dist < self.hole_radius:
            reward += 100
            done = True
            success = True

        if np.all(self.ball_vel == 0) and self.has_struck:
            reward -= 1
            done = True

        if self.steps > 200:
            done = True

        return self._get_obs(), reward, done, False, {"success": success}
