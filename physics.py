import numpy as np

class VectorPhysicsEngine:
    def __init__(self, num_envs=100, width=600, height=600):
        self.num_envs = num_envs
        self.width = width
        self.height = height
        
        # State tensors (N, 2)
        self.ball_pos = np.zeros((num_envs, 2))
        self.ball_vel = np.zeros((num_envs, 2))
        self.hole_pos = np.zeros((num_envs, 2))
        
        # Constants
        self.hole_radius = 0.04
        self.ball_radius = 0.015
        self.friction_coeff = 1.5
        self.stop_threshold = 0.05
        self.max_power = 2.5
        
        # Status
        self.active = np.ones(num_envs, dtype=bool)
        self.succeeded = np.zeros(num_envs, dtype=bool)
        self.final_dist = np.zeros(num_envs)

    def reset(self, ball_pos, hole_pos):
        # Broadcast single pos to all envs
        self.ball_pos[:] = ball_pos
        self.hole_pos[:] = hole_pos
        self.ball_vel[:] = 0
        self.active[:] = True
        self.succeeded[:] = False
        return self.get_state()

    def get_state(self):
        # Return (N, 2) relative vectors
        return self.ball_pos - self.hole_pos

    def hit_ball(self, angles, powers):
        # angles: (N,), powers: (N,)
        p = np.clip(powers, 0, 1.0) * self.max_power
        
        vx = np.cos(angles) * p
        vy = np.sin(angles) * p
        self.ball_vel = np.stack([vx, vy], axis=1)

    def step(self, dt=1/60.0):
        # Only update active balls
        # To keep it vectorized and simple without masking hell, we update all, 
        # but zero out velocity for inactive ones? 
        # Actually simpler to just update all and check stop conditions.
        
        # Friction
        speeds = np.linalg.norm(self.ball_vel, axis=1)
        moving = speeds > 0
        
        # Calculate drops
        drops = self.friction_coeff * dt
        
        # Apply friction where moving
        # If drop >= speed, stop. Else, slow down.
        # indices where we stop
        stop_mask = (drops >= speeds) & moving
        slow_mask = (drops < speeds) & moving
        
        self.ball_vel[stop_mask] = 0
        
        # self.ball_vel[slow_mask] -= (self.ball_vel[slow_mask] / speeds[slow_mask, None]) * drops
        # Vectorized safe division
        factors = np.zeros_like(speeds)
        factors[slow_mask] = drops / speeds[slow_mask]
        self.ball_vel -= self.ball_vel * factors[:, None]

        # Update position
        self.ball_pos += self.ball_vel * dt
        
        # Wall collisions
        # x < radius
        left_mask = self.ball_pos[:, 0] < self.ball_radius
        self.ball_pos[left_mask, 0] = self.ball_radius
        self.ball_vel[left_mask, 0] *= -0.7
        
        # x > 1 - radius
        right_mask = self.ball_pos[:, 0] > 1.0 - self.ball_radius
        self.ball_pos[right_mask, 0] = 1.0 - self.ball_radius
        self.ball_vel[right_mask, 0] *= -0.7
        
        # y < radius
        top_mask = self.ball_pos[:, 1] < self.ball_radius
        self.ball_pos[top_mask, 1] = self.ball_radius
        self.ball_vel[top_mask, 1] *= -0.7
        
        # y > 1 - radius
        bot_mask = self.ball_pos[:, 1] > 1.0 - self.ball_radius
        self.ball_pos[bot_mask, 1] = 1.0 - self.ball_radius
        self.ball_vel[bot_mask, 1] *= -0.7
        
        # Check hole
        dists = np.linalg.norm(self.ball_pos - self.hole_pos, axis=1)
        in_hole = dists < self.hole_radius
        
        # Mark succeeded
        # If just succeeded this step, we stop them
        new_success = in_hole & self.active
        self.succeeded[new_success] = True
        self.ball_vel[new_success] = 0 
        self.active[new_success] = False
        
        # Check stopped
        # If speed is low, stop
        new_stopped = (np.linalg.norm(self.ball_vel, axis=1) < self.stop_threshold) & self.active
        self.active[new_stopped] = False # No longer moving
        
        self.final_dist = dists
        
        # Return True if ALL are inactive (stopped or in hole)
        all_done = not np.any(self.active)
        return all_done, self.succeeded, dists
