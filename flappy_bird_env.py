import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time

# GAME CONSTANTS (matched to original Flappy Bird)
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
FLAP_POWER = -9.0        # upward impulse on flap
GRAVITY = 1.0             # downward acceleration per tick
MAX_FALL_SPEED = 10.0     # terminal velocity
PIPE_SPEED = 4.0          # horizontal pipe scroll per tick
PIPE_WIDTH = 52
PIPE_HEIGHT = 320
PIPE_GAP = 100
PIPE_SPACING = 200        # horizontal distance between pipe pairs
GROUND_HEIGHT = 112
BIRD_WIDTH = 34
BIRD_HEIGHT = 24

# REWARD SHAPING
REWARD_TICK        = 0.1   # reward given each tick the bird survives
REWARD_DEATH       = -1.0  # reward (penalty) on collision / out-of-bounds
REWARD_PIPE        = 1.5   # bonus reward for passing a pipe

# PIPE SPAWN BOUNDS (fraction of playable height)
PIPE_GAP_MIN_FRAC  = 0.2   # minimum gap-centre as fraction of playable height
PIPE_GAP_MAX_FRAC  = 0.6   # maximum gap-centre as fraction of playable height
PIPE_SPAWN_OFFSET  = 100   # extra pixels beyond screen edge before first pipe spawns

# RENDERING
SCORE_FONT_SIZE    = 36    # font size for the in-game score display

# Physics tick rate — game logic always runs at this rate
TICK_RATE = 30
TICK_DT = 1.0 / TICK_RATE


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Actions: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation: [bird_y, bird_speed, dist_to_next_pipe, top_pipe_bottom_y, bottom_pipe_top_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # Game state fields
        self.bird_x = 0.0
        self.bird_y = 0.0
        self.bird_speed = 0.0
        self.bird_dead = False
        self.score = 0
        self.frame = 0
        self.pipes = []
        self.ground_x = 0.0

        # Timing for frame-rate-independent rendering
        self.last_tick_time = None

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
            self.bird_speed / MAX_FALL_SPEED,
            dist_to_pipe,
            top_pipe_bottom,
            bot_pipe_top,
        ], dtype=np.float32)

        return obs

    def _spawn_pipe(self, x):
        # Gap can appear between PIPE_GAP_MIN_FRAC and PIPE_GAP_MAX_FRAC of playable height
        playable_height = SCREEN_HEIGHT - GROUND_HEIGHT
        min_gap_y = int(playable_height * PIPE_GAP_MIN_FRAC)
        max_gap_y = int(playable_height * PIPE_GAP_MAX_FRAC)
        gap_y = self.np_random.integers(min_gap_y, max_gap_y)
        self.pipes.append({"x": float(x), "gap_y": int(gap_y), "scored": False})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bird_x = SCREEN_WIDTH / 5
        self.bird_y = SCREEN_HEIGHT / 2
        self.bird_speed = 0.0
        self.bird_dead = False
        self.score = 0
        self.frame = 0

        self.pipes = []
        self._spawn_pipe(SCREEN_WIDTH + PIPE_SPAWN_OFFSET)
        self._spawn_pipe(SCREEN_WIDTH + PIPE_SPAWN_OFFSET + PIPE_SPACING)

        self.ground_x = 0.0
        self.last_tick_time = time.perf_counter()

        obs = self._get_obs()
        return obs, {}

    def _physics_tick(self, action):
        """Advance game state by exactly one fixed-rate physics tick."""
        self.frame += 1

        # Apply action
        if action == 1 and not self.bird_dead:
            self.bird_speed = FLAP_POWER

        # Update bird physics
        self.bird_speed += GRAVITY
        self.bird_speed = min(self.bird_speed, MAX_FALL_SPEED)
        self.bird_y += self.bird_speed

        # Update pipes
        for p in self.pipes:
            p["x"] -= PIPE_SPEED

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if p["x"] + PIPE_WIDTH > 0]

        # Spawn new pipes to maintain at least 2 ahead
        if len(self.pipes) < 2:
            rightmost = max(p["x"] for p in self.pipes) if self.pipes else SCREEN_WIDTH
            spacing = max(rightmost + PIPE_SPACING, SCREEN_WIDTH + 100)
            self._spawn_pipe(spacing)

        # Update ground scroll
        self.ground_x = (self.ground_x - PIPE_SPEED) % -(SCREEN_WIDTH * 2)

        # Bird rect for collision
        bird_rect = pygame.Rect(self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT)

        # Check collisions
        terminated = False
        reward = REWARD_TICK  # small reward for surviving each tick

        # Ceiling
        if self.bird_y <= 0:
            terminated = True

        # Ground
        if self.bird_y + BIRD_HEIGHT >= SCREEN_HEIGHT - GROUND_HEIGHT:
            terminated = True

        # Pipe collision
        for p in self.pipes:
            top_rect = pygame.Rect(p["x"], p["gap_y"] - PIPE_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT)
            bot_rect = pygame.Rect(p["x"], p["gap_y"] + PIPE_GAP, PIPE_WIDTH, PIPE_HEIGHT)
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bot_rect):
                terminated = True

        if terminated:
            reward = REWARD_DEATH

        # Score: flag-based, triggers exactly once per pipe
        bird_right = self.bird_x + BIRD_WIDTH
        for p in self.pipes:
            pipe_right = p["x"] + PIPE_WIDTH
            if not p["scored"] and bird_right > pipe_right:
                p["scored"] = True
                self.score += 1
                if not terminated:
                    reward += REWARD_PIPE

        obs = self._get_obs()
        truncated = False

        return obs, reward, terminated, truncated, {"score": self.score}

    def step(self, action):
        # Always run exactly one physics tick per step call.
        # This keeps training deterministic: one action = one tick.
        obs, reward, terminated, truncated, info = self._physics_tick(action)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

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
            bot_y = p["gap_y"] + PIPE_GAP
            self.screen.blit(self.pipe_img, (p["x"], bot_y))
            top_y = p["gap_y"] - PIPE_HEIGHT
            self.screen.blit(self.pipe_img_flipped, (p["x"], top_y))

        # Draw ground
        self.screen.blit(self.ground_img, (self.ground_x, SCREEN_HEIGHT - GROUND_HEIGHT))
        self.screen.blit(self.ground_img, (self.ground_x + SCREEN_WIDTH * 2, SCREEN_HEIGHT - GROUND_HEIGHT))

        # Draw bird
        self.screen.blit(self.bird_img, (self.bird_x, self.bird_y))

        # Draw collision boxes (debug)
        pygame.draw.rect(self.screen, (255, 0, 0), (self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT), 2)
        for p in self.pipes:
            top_rect = pygame.Rect(p["x"], p["gap_y"] - PIPE_HEIGHT, PIPE_WIDTH, PIPE_HEIGHT)
            bot_rect = pygame.Rect(p["x"], p["gap_y"] + PIPE_GAP, PIPE_WIDTH, PIPE_HEIGHT)
            pygame.draw.rect(self.screen, (0, 255, 0), top_rect, 2)
            pygame.draw.rect(self.screen, (0, 255, 0), bot_rect, 2)

        # Draw score
        font = pygame.font.SysFont(None, SCORE_FONT_SIZE)
        score_surf = font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_surf, (SCREEN_WIDTH / 2, 30))

        pygame.display.flip()

        # Cap rendering FPS without affecting physics
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