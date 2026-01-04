import pygame
import numpy as np

WIDTH, HEIGHT = 1280, 720
BG_COLOR = (34, 139, 34)
HOLE_COLOR = (20, 20, 20)
BALL_COLOR = (255, 255, 255)
UI_BG_COLOR = (0, 0, 0, 150)
TEXT_COLOR = (255, 255, 255)
SLIDER_BG = (50, 50, 50)
SLIDER_FG = (200, 200, 200)

class SimpleSlider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.dragging = False
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.update_val(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.update_val(event.pos[0])
                
    def update_val(self, mouse_x):
        ratio = (mouse_x - self.rect.x) / self.rect.width
        ratio = np.clip(ratio, 0, 1)
        self.val = self.min_val + (self.max_val - self.min_val) * ratio
        
    def draw(self, screen):
        pygame.draw.rect(screen, SLIDER_BG, self.rect)
        
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + ratio * self.rect.width
        handle_rect = pygame.Rect(handle_x - 5, self.rect.y - 5, 10, self.rect.height + 10)
        pygame.draw.rect(screen, SLIDER_FG, handle_rect)
        
        font = pygame.font.SysFont("Arial", 16)
        text = font.render(f"Speed: {int(self.val)}x", True, TEXT_COLOR)
        screen.blit(text, (self.rect.x, self.rect.y - 20))
        
    def get_value(self):
        return self.val

class GolfUI:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 20)
        self.header_font = pygame.font.SysFont("Arial", 30, bold=True)
        self.slider = SimpleSlider(WIDTH - 160, 20, 140, 10, 1, 500, 1)
        
    def draw_board(self, env):
        self.screen.fill(BG_COLOR)
        
        for x in range(0, WIDTH, 100):
            pygame.draw.line(self.screen, (30, 120, 30), (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 100):
            pygame.draw.line(self.screen, (30, 120, 30), (0, y), (WIDTH, y), 1)

        # Draw Barriers
        for b in env.barriers:
            # b is [x, y, w, h] normalized
            rect = pygame.Rect(
                int(b[0]*WIDTH), int(b[1]*HEIGHT),
                int(b[2]*WIDTH), int(b[3]*HEIGHT)
            )
            pygame.draw.rect(self.screen, (139, 69, 19), rect) # Brown wood color
            pygame.draw.rect(self.screen, (100, 50, 10), rect, 2) # Darker border

        hx, hy = env.hole_pos[0][0] * WIDTH, env.hole_pos[0][1] * HEIGHT
        r = env.hole_radius * WIDTH
        pygame.draw.circle(self.screen, (50, 50, 50), (int(hx), int(hy)), int(r) + 2)
        pygame.draw.circle(self.screen, HOLE_COLOR, (int(hx), int(hy)), int(r))
        
    def draw_balls(self, env):
        if not (env.active[0] or env.succeeded[0]):
             return

        bx, by = env.ball_pos[0][0] * WIDTH, env.ball_pos[0][1] * HEIGHT
        br = env.ball_radius * WIDTH
        
        pygame.draw.circle(self.screen, (0, 0, 0, 80), (int(bx)+3, int(by)+3), int(br))
        
        pygame.draw.circle(self.screen, (240, 240, 240), (int(bx), int(by)), int(br))
        
        pygame.draw.circle(self.screen, (200, 200, 200), (int(bx), int(by)), int(br), width=1)
        
        pygame.draw.circle(self.screen, (255, 255, 255), (int(bx)-int(br*0.3), int(by)-int(br*0.3)), int(br*0.3))

    def draw_setup_ball(self, pos, ball_radius):
        bx, by = pos[0] * WIDTH, pos[1] * HEIGHT
        br = ball_radius * WIDTH
        pygame.draw.circle(self.screen, (0, 50, 0), (int(bx)+2, int(by)+2), int(br))
        pygame.draw.circle(self.screen, BALL_COLOR, (int(bx), int(by)), int(br))

    def draw_trail(self, trail, color=(255, 255, 255), width=2):
        if len(trail) < 2:
            return
        
        points = [(int(p[0]*WIDTH), int(p[1]*HEIGHT)) for p in trail]
        pygame.draw.lines(self.screen, color, False, points, width)

    def draw_hud(self, mode, episode, reward, last_dist, success_rate):
        panel_rect = pygame.Rect(10, 10, 330, 160)
        s = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        s.fill(UI_BG_COLOR)
        self.screen.blit(s, panel_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), panel_rect, 2)
        
        y_off = 20
        title = self.header_font.render(f"RL based Golf Simulation", True, (255, 215, 0))
        self.screen.blit(title, (30, y_off))
        y_off += 40
        
        lines = [
            f"Episode: {episode}",
            f"Avg Reward: {reward:.2f}",
            f"Avg Dist: {last_dist:.3f}",
            f"Success Rate: {success_rate*100:.1f}%"
        ]
        
        for line in lines:
            t = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(t, (30, y_off))
            y_off += 25
            
        self.slider.draw(self.screen)

    def draw_setup_instructions(self):
        panel_rect = pygame.Rect(10, HEIGHT - 60, WIDTH - 20, 50)
        s = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        s.fill(UI_BG_COLOR)
        self.screen.blit(s, panel_rect)
        
        text = self.font.render("Instruction: [Left] Ball | [Right] Hole | [Shift+Left] Add Barrier | [Shift+Right] Remove Barrier", True, TEXT_COLOR)
        text_rect = text.get_rect(center=(WIDTH//2, HEIGHT - 35))
        self.screen.blit(text, text_rect)
        
        self.slider.draw(self.screen)
