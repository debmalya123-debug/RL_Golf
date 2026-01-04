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

    # Vectorized Environment: 100 parallel balls
    NUM_ENVS = 100
    env = Env(num_envs=NUM_ENVS, width=WIDTH, height=HEIGHT)
    
    ui = GolfUI(screen)
    agent = PPOAgent(state_dim=2, action_dim=2, lr=5e-4, gamma=0.90) 

    # Interactive State
    mode = "setup" 
    user_ball_pos = np.array([0.2, 0.5])
    user_hole_pos = np.array([0.8, 0.5])
    
    # Training Stats
    episode = 0
    BATCH_SIZE = 500 # minimum transitions before update
    last_avg_reward = 0.0
    last_avg_dist = 0.0
    success_history = [] # Global history
    
    # State tracking
    is_rolling = False
    
    # Temp storage for the current batch shot
    current_states = None
    current_actions = None
    current_logprobs = None
    
    # Visualization trail (just for env 0)
    current_trail = []

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
                        norm_pos = np.array([mx/WIDTH, my/HEIGHT])
                        if event.button == 1: 
                            user_ball_pos = norm_pos
                            # Preview positions
                            env.ball_pos[:] = user_ball_pos
                        elif event.button == 3: 
                            user_hole_pos = norm_pos
                            env.hole_pos[:] = user_hole_pos
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        mode = "train"
                        # Reset all envs
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

        # Loop Logic
        if mode == "train":
            # Speed Control: How many simulation steps per frame?
            steps_per_frame = int(ui.slider.get_value())
            
            for _ in range(steps_per_frame):
                if not is_rolling:
                    # 1. OBSERVATION
                    current_states = env.get_state() # (N, 2)
                    
                    # 2. ACTION
                    actions, logprobs, _ = agent.select_action(current_states)
                    
                    # Store for update
                    current_actions = actions
                    current_logprobs = logprobs
                    
                    # Apply Action
                    angles = actions[:, 0]
                    powers = np.abs(actions[:, 1])
                    env.hit_ball(angles, powers)
                    
                    is_rolling = True
                    current_step_count = 0
                    current_trail = [env.ball_pos[0].copy()] # Track env 0
                
                else:
                    # 3. PHYSICS STEP (Vectorized)
                    all_done, succeeded_mask, dists = env.step(dt)
                    current_step_count += 1
                    
                    # Track env 0 for visuals
                    if current_step_count % 5 == 0:
                        current_trail.append(env.ball_pos[0].copy())
                    
                    # Check if all stopped or timeout
                    if all_done or current_step_count > 300:
                        is_rolling = False
                        
                        # 4. REWARD CALCULATION (Batch)
                        # Distance penalty + Success Bonus
                        # New Shaping: Success (+10), Fail (-Distance * 10)
                        # This balances the magnitudes.
                        rewards = -dists * 10.0
                        rewards[succeeded_mask] += 20.0 # Big bonus
                        
                        # Stats
                        num_success = np.sum(succeeded_mask)
                        batch_success_rate = num_success / NUM_ENVS
                        success_history.append(batch_success_rate)
                        if len(success_history) > 50:
                            success_history.pop(0)
                            
                        last_avg_reward = np.mean(rewards)
                        last_avg_dist = np.mean(dists)
                        episode += 1
                        
                        # 5. RESTORE TO AGENT
                        # We gathered NUM_ENVS transitions from this shot
                        agent.store(current_states, current_actions, current_logprobs, rewards)
                        
                        # 6. UPDATE IF READY
                        # We check total stored items. Since store() appends lists of arrays,
                        # we check length of lists * NUM_ENVS
                        total_samples = len(agent.states) * NUM_ENVS
                        if total_samples >= BATCH_SIZE:
                            agent.update()
                            print(f"Update @ Ep {episode} | Avg Reward: {last_avg_reward:.2f} | Success: {batch_success_rate*100:.1f}%")
                        
                        # Reset for next shot
                        env.reset(user_ball_pos, user_hole_pos)
                        # Break inner loop to render updated state
                        break

        # Render
        ui.draw_board(env)
        if mode == "train":
            ui.draw_balls(env) # Draw ghost clouds
            ui.draw_trail(current_trail) # Draw leader trail
            
            sr = sum(success_history) / len(success_history) if success_history else 0.0
            ui.draw_hud(mode, episode, last_avg_reward, last_avg_dist, sr)
        else:
             ui.draw_setup_ball(user_ball_pos, env.ball_radius)
             ui.draw_setup_instructions()
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
