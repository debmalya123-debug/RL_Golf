import pygame
import numpy as np
from physics import VectorPhysicsEngine as Env 
from ui import GolfUI, WIDTH, HEIGHT
from agent import PPOAgent

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Based Golf Simulation")
    clock = pygame.time.Clock()

    NUM_ENVS = 500
    env = Env(num_envs=NUM_ENVS, width=WIDTH, height=HEIGHT)
    
    ui = GolfUI(screen)
    # State Dim is now 10: 2 (Target) + 8 (Lidar)
    agent = PPOAgent(state_dim=10, action_dim=2, lr=5e-4, gamma=0.90) 

    mode = "setup" 
    user_ball_pos = np.array([0.2, 0.5])
    user_hole_pos = np.array([0.8, 0.5])
    
    episode = 0
    BATCH_SIZE = 500
    last_avg_reward = 0.0
    last_avg_dist = 0.0
    success_history = [] 
    
    is_rolling = False
    
    current_states = None
    current_actions = None
    current_logprobs = None
    
    current_trail = []
    
    drag_start = None
    curr_drag = None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            ui.slider.handle_event(event)

            if mode == "setup":
                if event.type == pygame.MOUSEBUTTONDOWN and not ui.slider.dragging:
                    mx, my = event.pos
                    if not ui.slider.rect.inflate(20, 20).collidepoint((mx, my)):
                        keys = pygame.key.get_pressed()
                        norm_pos = np.array([mx/WIDTH, my/HEIGHT])
                        
                        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                            if event.button == 1: 
                                drag_start = norm_pos
                                curr_drag = [norm_pos[0], norm_pos[1], 0, 0]
                            elif event.button == 3: 
                                env.remove_barrier(norm_pos)
                        else:
                            if event.button == 1: 
                                user_ball_pos = norm_pos
                                env.ball_pos[:] = user_ball_pos
                            elif event.button == 3: 
                                user_hole_pos = norm_pos
                                env.hole_pos[:] = user_hole_pos
                
                if event.type == pygame.MOUSEBUTTONUP:
                    if drag_start is not None:
                        x, y, w, h = curr_drag
                        if w < 0: x += w; w = -w
                        if h < 0: y += h; h = -h
                        
                        if w > 0.01 and h > 0.01:
                            env.add_barrier([x, y, w, h])
                            
                        drag_start = None
                        curr_drag = None

                if event.type == pygame.MOUSEMOTION:
                    if drag_start is not None:
                         norm_pos = np.array([event.pos[0]/WIDTH, event.pos[1]/HEIGHT])
                         curr_drag[2] = norm_pos[0] - drag_start[0]
                         curr_drag[3] = norm_pos[1] - drag_start[1]
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        mode = "train"
                        env.reset(user_ball_pos, user_hole_pos)
                        is_rolling = False
                        current_trail = []
                        episode = 0
                        
            elif mode == "train":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        mode = "setup"
                        is_rolling = False
                        env.ball_vel[:] = 0

        if mode == "train":
            steps_per_frame = int(ui.slider.get_value())
            
            for _ in range(steps_per_frame):
                if not is_rolling:
                    current_states = env.get_state()
                    
                    actions, logprobs, _ = agent.select_action(current_states)
                    
                    current_actions = actions
                    current_logprobs = logprobs
                    
                    angles = actions[:, 0]
                    powers = np.abs(actions[:, 1])
                    env.hit_ball(angles, powers)
                    
                    is_rolling = True
                    current_step_count = 0
                    current_trail = [env.ball_pos[0].copy()]
                
                else:
                    all_done, succeeded_mask, dists = env.step(dt)
                    current_step_count += 1
                    
                    if current_step_count % 5 == 0:
                        current_trail.append(env.ball_pos[0].copy())
                    
                    if all_done or current_step_count > 300:
                        is_rolling = False
                        
                        rewards = -dists * 10.0
                        rewards[succeeded_mask] += 20.0
                        
                        num_success = np.sum(succeeded_mask)
                        batch_success_rate = num_success / NUM_ENVS
                        success_history.append(batch_success_rate)
                        if len(success_history) > 50:
                            success_history.pop(0)
                            
                        last_avg_reward = np.mean(rewards)
                        last_avg_dist = np.mean(dists)
                        episode += 1
                        
                        agent.store(current_states, current_actions, current_logprobs, rewards)
                        
                        total_samples = len(agent.states) * NUM_ENVS
                        if total_samples >= BATCH_SIZE:
                            agent.update()
                            print(f"Update @ Ep {episode} | Avg Reward: {last_avg_reward:.2f} | Success: {batch_success_rate*100:.1f}%")
                        
                        env.reset(user_ball_pos, user_hole_pos)
                        break

        ui.draw_board(env)
        if mode == "train":
            ui.draw_balls(env) 
            ui.draw_trail(current_trail) 
            
            sr = sum(success_history) / len(success_history) if success_history else 0.0
            ui.draw_hud(mode, episode, last_avg_reward, last_avg_dist, sr)
        else:
             ui.draw_setup_ball(user_ball_pos, env.ball_radius)
             ui.draw_setup_instructions()
             if curr_drag is not None:
                 r = curr_drag[:]
                 if r[2] < 0: r[0] += r[2]; r[2] = -r[2]
                 if r[3] < 0: r[1] += r[3]; r[3] = -r[3]
                 
                 rect = pygame.Rect(
                    int(r[0]*WIDTH), int(r[1]*HEIGHT),
                    int(r[2]*WIDTH), int(r[3]*HEIGHT)
                 )
                 pygame.draw.rect(screen, (200, 100, 50), rect, 2)
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
