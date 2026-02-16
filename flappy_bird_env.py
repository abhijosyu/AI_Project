import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

# GAME CONSTANTS
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 8
GRAVITY = 0.5
GAME_SPEED = 3
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150
GROUND_HEIGHT = 100

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Actions: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation: [bird_y, bird_speed, dist_to_next_pipe, top_pipe_bottom_y, bottom_pipe_top_y]
        # All values normalized to roughly [0, 1] range
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Pygame setup (only init display if rendering)
        self.screen = None
        self.clock = None
        self.background = None

        if self.render_mode == "human":
            self._init_render()

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()
        self.background = pygame.image.load('assets/sprites/background-day.png')
        self.background = pygame.transform.scale(self.background, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.bird_img = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.pipe_img = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.pipe_img = pygame.transform.scale(self.pipe_img, (PIPE_WIDTH, PIPE_HEIGHT))
        self.pipe_img_flipped = pygame.transform.flip(self.pipe_img, False, True)
        self.ground_img = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.ground_img = pygame.transform.scale(self.ground_img, (SCREEN_WIDTH * 2, GROUND_HEIGHT))

    def _get_obs(self):
        # Find the next pipe pair (closest pipe ahead of the bird)
        next_pipe = None
        min_dist = float('inf')
        for p in self.pipes:
            right_edge = p["x"] + PIPE_WIDTH
            if right_edge > self.bird_x:
                dist = p["x"] - self.bird_x
                if dist < min_dist:
                    min_dist = dist
                    next_pipe = p

        if next_pipe is None:
            # No pipes ahead, use defaults
            dist_to_pipe = 1.0
            top_pipe_bottom = 0.5
            bot_pipe_top = 0.5
        else:
            dist_to_pipe = (next_pipe["x"] - self.bird_x) / SCREEN_WIDTH
            gap_top = next_pipe["gap_y"]
            gap_bot = next_pipe["gap_y"] + PIPE_GAP
            top_pipe_bottom = gap_top / SCREEN_HEIGHT
            bot_pipe_top = gap_bot / SCREEN_HEIGHT

        obs = np.array([
            self.bird_y / SCREEN_HEIGHT,
            self.bird_speed / SPEED,
            dist_to_pipe,
            top_pipe_bottom,
            bot_pipe_top,
        ], dtype=np.float32)

        return obs

    def _spawn_pipe(self, x):
        gap_y = random.randint(100, SCREEN_HEIGHT - GROUND_HEIGHT - PIPE_GAP - 100)
        self.pipes.append({"x": x, "gap_y": gap_y})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bird_x = SCREEN_WIDTH / 6
        self.bird_y = SCREEN_HEIGHT / 2
        self.bird_speed = 0
        self.bird_dead = False
        self.score = 0
        self.frame = 0

        # Pipe list: each pipe is {"x": float, "gap_y": float}
        self.pipes = []
        self._spawn_pipe(SCREEN_WIDTH + 100)
        self._spawn_pipe(SCREEN_WIDTH + 100 + 300)

        self.ground_x = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.frame += 1

        # Apply action
        if action == 1 and not self.bird_dead:
            self.bird_speed = -SPEED

        # Update bird physics
        self.bird_speed += GRAVITY
        self.bird_y += self.bird_speed

        # Update pipes
        for p in self.pipes:
            p["x"] -= GAME_SPEED

        # Remove off-screen pipes and score
        new_pipes = []
        for p in self.pipes:
            if p["x"] + PIPE_WIDTH > 0:
                new_pipes.append(p)
            else:
                self.score += 1
        self.pipes = new_pipes

        # Spawn new pipes
        if len(self.pipes) < 2:
            rightmost = max(p["x"] for p in self.pipes) if self.pipes else SCREEN_WIDTH
            spacing = max(rightmost + 300, SCREEN_WIDTH + 100)
            self._spawn_pipe(spacing)

        # Update ground scroll
        self.ground_x = (self.ground_x - GAME_SPEED) % -(SCREEN_WIDTH * 2)

        # Bird rect for collision
        bird_w, bird_h = 34, 24  # approximate bird sprite size
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, bird_w, bird_h)

        # Check collisions
        terminated = False
        reward = 0.1  # small reward for surviving each frame

        # Ceiling
        if self.bird_y <= 0:
            terminated = True

        # Ground
        if self.bird_y + bird_h >= SCREEN_HEIGHT - GROUND_HEIGHT:
            terminated = True

        # Pipes
        for p in self.pipes:
            top_rect = pygame.Rect(p["x"], p["gap_y"] - PIPE_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT)
            bot_rect = pygame.Rect(p["x"], p["gap_y"] + PIPE_GAP, PIPE_WIDTH, PIPE_HEIGHT)
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bot_rect):
                terminated = True

        if terminated:
            reward = -1.0

        # Bonus reward for passing a pipe
        for p in self.pipes:
            pipe_mid = p["x"] + PIPE_WIDTH / 2
            bird_mid = self.bird_x + bird_w / 2
            if abs(pipe_mid - bird_mid) < GAME_SPEED / 2 + 0.5:
                reward += 1.0

        obs = self._get_obs()
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"score": self.score}

    def render(self):
        if self.screen is None:
            self._init_render()

        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.blit(self.background, (0, 0))

        # Draw pipes
        for p in self.pipes:
            # Bottom pipe
            bot_y = p["gap_y"] + PIPE_GAP
            self.screen.blit(self.pipe_img, (p["x"], bot_y))
            # Top pipe (flipped)
            top_y = p["gap_y"] - PIPE_HEIGHT
            self.screen.blit(self.pipe_img_flipped, (p["x"], top_y))

        # Draw ground
        self.screen.blit(self.ground_img, (self.ground_x, SCREEN_HEIGHT - GROUND_HEIGHT))
        self.screen.blit(self.ground_img, (self.ground_x + SCREEN_WIDTH * 2, SCREEN_HEIGHT - GROUND_HEIGHT))

        # Draw bird
        self.screen.blit(self.bird_img, (self.bird_x, self.bird_y))

        # Draw score
        font = pygame.font.SysFont(None, 36)
        score_surf = font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_surf, (SCREEN_WIDTH / 2, 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ---- Manual play mode ----
if __name__ == "__main__":
    env = FlappyBirdEnv(render_mode="human")
    obs, info = env.reset()
    running = True

    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
                if event.key == pygame.K_r:
                    obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"Game Over! Score: {info['score']}. Press R to restart")
            # Wait for restart
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            obs, info = env.reset()
                            waiting = False

    env.close()