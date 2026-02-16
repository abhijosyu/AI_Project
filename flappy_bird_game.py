import pygame, random

# VARIABLES
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 8
GRAVITY = 0.5
GAME_SPEED = 3

GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT= 100

PIPE_WIDTH = 80
PIPE_HEIGHT = 500

PIPE_GAP = 150

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images =  [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]

        self.speed = 0 
        self.dead = False 

        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.speed += GRAVITY

        #UPDATE HEIGHT
        self.rect[1] += self.speed

    def bump(self):
        if not self.dead:  # Only allow bumping if not dead
            self.speed = -SPEED

    def die(self):
        self.dead = True

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self. image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))


        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize


        self.mask = pygame.mask.from_surface(self.image)


    def update(self):
        self.rect[0] -= GAME_SPEED

class Ground(pygame.sprite.Sprite):
    
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT
    def update(self):
        self.rect[0] -= GAME_SPEED


def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')
BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))

def create_ground():
    ground_group = pygame.sprite.Group()
    ground_group.add(Ground(0))
    ground_group.add(Ground(GROUND_WIDTH))
    return ground_group

def reset_game():
    bird = Bird()
    bird_group = pygame.sprite.Group()
    bird_group.add(bird)

    pipe_group = pygame.sprite.Group()

    ground_group = create_ground()
    score = 0

    return bird, bird_group, pipe_group, ground_group, score

bird, bird_group, pipe_group, ground_group, score = reset_game()
game_over = False

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird.bump()
            if event.key == pygame.K_r and game_over:
                bird, bird_group, pipe_group, ground_group, score = reset_game()
                game_over = False

    # UPDATE
    bird_group.update()
    
    if not bird.dead:
        pipe_group.update()
        ground_group.update()

        # Loop ground segments: when one scrolls off-screen, move it to the right
        for g in ground_group:
            if is_off_screen(g):
                g.rect[0] = GROUND_WIDTH

        # Spawn new pipes when old ones scroll off-screen
        for p in list(pipe_group):
            if is_off_screen(p):
                pipe_group.remove(p)
                score += 0.5  # Each pipe pair has 2 sprites, so 0.5 each = 1 point per pair
        
        if len(pipe_group) < 4:
            # Find the rightmost pipe to place the new pair after it
            rightmost = max(p.rect[0] for p in pipe_group) if pipe_group else SCREEN_WIDTH
            spacing = max(rightmost + 300, SCREEN_WIDTH + 100)
            new_pipes = get_random_pipes(spacing)
            pipe_group.add(new_pipes[0])
            pipe_group.add(new_pipes[1])

    # CHECK COLLISIONS
    if not bird.dead:
        # Check if bird hits pipes
        if pygame.sprite.spritecollide(bird, pipe_group, False, pygame.sprite.collide_mask):
            bird.die()
        
        # Check if bird hits ground
        if pygame.sprite.spritecollide(bird, ground_group, False, pygame.sprite.collide_mask):
            bird.die()

        # Check if bird hits ceiling
        if bird.rect[1] <= 0:
            bird.die()
    
    # When dead, stop bird at ground level
    if bird.dead and bird.rect[1] >= SCREEN_HEIGHT - GROUND_HEIGHT - bird.rect[3]:
        bird.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT - bird.rect[3]
        bird.speed = 0
        if not game_over:
            print(f"Game Over! Score: {int(score)}. Press R to restart")
            game_over = True

    # RENDER
    screen.blit(BACKGROUND, (0, 0))
    pipe_group.draw(screen)
    ground_group.draw(screen)
    bird_group.draw(screen)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()