import numpy as np

class VectorPhysicsEngine:
    def __init__(self, num_envs=100, width=600, height=600):
        self.num_envs = num_envs
        self.width = width
        self.height = height
        
        self.ball_pos = np.zeros((num_envs, 2))
        self.ball_vel = np.zeros((num_envs, 2))
        self.hole_pos = np.zeros((num_envs, 2))
        
        self.hole_radius = 0.04
        self.ball_radius = 0.015
        self.friction_coeff = 1.5
        self.stop_threshold = 0.05
        self.max_power = 2.5
        
        self.active = np.ones(num_envs, dtype=bool)
        self.succeeded = np.zeros(num_envs, dtype=bool)
        self.final_dist = np.zeros(num_envs)

    def reset(self, ball_pos, hole_pos):
        self.ball_pos[:] = ball_pos
        self.hole_pos[:] = hole_pos
        self.ball_vel[:] = 0
        self.active[:] = True
        self.succeeded[:] = False
        return self.get_state()

    def get_state(self):
        return self.ball_pos - self.hole_pos

    def hit_ball(self, angles, powers):
        p = np.clip(powers, 0, 1.0) * self.max_power
        
        vx = np.cos(angles) * p
        vy = np.sin(angles) * p
        self.ball_vel = np.stack([vx, vy], axis=1)

    def step(self, dt=1/60.0):
        speeds = np.linalg.norm(self.ball_vel, axis=1)
        moving = speeds > 0
        
        drops = self.friction_coeff * dt
        
        stop_mask = (drops >= speeds) & moving
        slow_mask = (drops < speeds) & moving
        
        self.ball_vel[stop_mask] = 0
        
        factors = np.zeros_like(speeds)
        factors[slow_mask] = drops / speeds[slow_mask]
        self.ball_vel -= self.ball_vel * factors[:, None]

        self.ball_pos += self.ball_vel * dt
        
        left_mask = self.ball_pos[:, 0] < self.ball_radius
        self.ball_pos[left_mask, 0] = self.ball_radius
        self.ball_vel[left_mask, 0] *= -0.7
        
        right_mask = self.ball_pos[:, 0] > 1.0 - self.ball_radius
        self.ball_pos[right_mask, 0] = 1.0 - self.ball_radius
        self.ball_vel[right_mask, 0] *= -0.7
        
        top_mask = self.ball_pos[:, 1] < self.ball_radius
        self.ball_pos[top_mask, 1] = self.ball_radius
        self.ball_vel[top_mask, 1] *= -0.7
        
        bot_mask = self.ball_pos[:, 1] > 1.0 - self.ball_radius
        self.ball_pos[bot_mask, 1] = 1.0 - self.ball_radius
        self.ball_vel[bot_mask, 1] *= -0.7
        
        dists = np.linalg.norm(self.ball_pos - self.hole_pos, axis=1)
        in_hole = dists < self.hole_radius
        
        new_success = in_hole & self.active
        self.succeeded[new_success] = True
        self.ball_vel[new_success] = 0 
        self.active[new_success] = False
        
        new_stopped = (np.linalg.norm(self.ball_vel, axis=1) < self.stop_threshold) & self.active
        self.active[new_stopped] = False 
        
        self.final_dist = dists
        
        all_done = not np.any(self.active)
        return all_done, self.succeeded, dists
