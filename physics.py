import numpy as np

class VectorPhysicsEngine:
    def __init__(self, num_envs=100, width=600, height=600):
        self.num_envs = num_envs
        self.width = width
        self.height = height
        
        self.ball_pos = np.zeros((num_envs, 2))
        self.ball_vel = np.zeros((num_envs, 2))
        self.hole_pos = np.zeros((num_envs, 2))
        
        self.barriers = [] 
        
        self.hole_radius = 0.04
        self.ball_radius = 0.015
        self.friction_coeff = 1.5
        self.stop_threshold = 0.05
        self.max_power = 2.5
        
        self.active = np.ones(num_envs, dtype=bool)
        self.succeeded = np.zeros(num_envs, dtype=bool)
        self.final_dist = np.zeros(num_envs)

        self.lidar_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        self.lidar_vecs = np.stack([np.cos(self.lidar_angles), np.sin(self.lidar_angles)], axis=1)

    def reset(self, ball_pos, hole_pos):
        # Add slight noise to start position for generalization (except env 0 for visuals)
        self.ball_pos[:] = ball_pos
        self.ball_pos[1:] += np.random.normal(0, 0.01, (self.num_envs-1, 2))
        
        self.hole_pos[:] = hole_pos
        self.ball_vel[:] = 0
        self.active[:] = True
        self.succeeded[:] = False
        return self.get_state()

    def add_barrier(self, rect):
        self.barriers.append(np.array(rect))
        
    def remove_barrier(self, pos):
        mx, my = pos
        to_remove = []
        for i, b in enumerate(self.barriers):
            bx, by, bw, bh = b
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                to_remove.append(i)
        
        for i in reversed(to_remove):
            self.barriers.pop(i)

    def get_state(self):
        rel_pos = self.ball_pos - self.hole_pos
        
        lidar = self._compute_lidar()
        
        return np.concatenate([rel_pos, lidar], axis=1)

    def _compute_lidar(self):
        dists = np.full((self.num_envs, 8), 10.0)
        
        for i, (dx, dy) in enumerate(self.lidar_vecs):
            if dx < 0:
                t = (0 - self.ball_pos[:, 0]) / dx
                dists[:, i] = np.minimum(dists[:, i], t)
            elif dx > 0:
                t = (1 - self.ball_pos[:, 0]) / dx
                dists[:, i] = np.minimum(dists[:, i], t)
                
            if dy < 0:
                t = (0 - self.ball_pos[:, 1]) / dy
                dists[:, i] = np.minimum(dists[:, i], t)
            elif dy > 0:
                t = (1 - self.ball_pos[:, 1]) / dy
                dists[:, i] = np.minimum(dists[:, i], t)
                
        for b in self.barriers:
            bx, by, bw, bh = b
            b_min = np.array([bx, by])
            b_max = np.array([bx+bw, by+bh])
            
            for i, (dx, dy) in enumerate(self.lidar_vecs):
                inv_dx = 1.0 / (dx + 1e-6)
                inv_dy = 1.0 / (dy + 1e-6)
                
                t1 = (b_min[0] - self.ball_pos[:, 0]) * inv_dx
                t2 = (b_max[0] - self.ball_pos[:, 0]) * inv_dx
                t3 = (b_min[1] - self.ball_pos[:, 1]) * inv_dy
                t4 = (b_max[1] - self.ball_pos[:, 1]) * inv_dy
                
                tmin = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
                tmax = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
                
                hit = (tmax >= tmin) & (tmax >= 0)
                
                mask = hit & (tmin < dists[:, i]) & (tmin > 0)
                dists[mask, i] = tmin[mask]
        
        return np.clip(dists, 0, 1.5)

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

        next_pos = self.ball_pos + self.ball_vel * dt
        
        rx = self.ball_radius
        ry = self.ball_radius * (self.width / self.height)
        
        hit_factor = 0.8
        pad_x = rx * hit_factor
        pad_y = ry * hit_factor
        
        for b in self.barriers:
            bx, by, bw, bh = b
            
            overlap_x = (next_pos[:, 0] > bx - pad_x) & (next_pos[:, 0] < bx + bw + pad_x)
            overlap_y = (next_pos[:, 1] > by - pad_y) & (next_pos[:, 1] < by + bh + pad_y)
            mask = overlap_x & overlap_y & self.active
            
            if np.any(mask):
                prev_x = self.ball_pos[mask, 0]
                prev_y = self.ball_pos[mask, 1]
                
                idxs = np.where(mask)[0]
                for idx in idxs:
                    px, py = self.ball_pos[idx]
                    vx, vy = self.ball_vel[idx]
                    
                    was_left = px < bx
                    was_right = px > bx + bw
                    was_top = py < by
                    was_bot = py > by + bh
                    
                    hit_x_side = was_left or was_right
                    hit_y_side = was_top or was_bot
                    
                    if hit_x_side and not hit_y_side:
                        vx = -vx * 0.7
                    elif hit_y_side and not hit_x_side:
                        vy = -vy * 0.7
                    elif hit_x_side and hit_y_side:
                        vx = -vx * 0.7
                        vy = -vy * 0.7
                    else:
                        if abs(vx) > abs(vy): vx = -vx * 0.7 
                        else: vy = -vy * 0.7
                        
                    self.ball_vel[idx] = [vx, vy]
                    next_pos[idx] = self.ball_pos[idx]

        self.ball_pos = next_pos
        
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
