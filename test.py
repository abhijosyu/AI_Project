from flappy_bird_env import FlappyBirdEnv
from gymnasium.utils.env_checker import check_env

env = FlappyBirdEnv(render_mode="human")

check_env(env)
print("Environment passed all checks!")

obs, _ = env.reset()

for episode in range(100):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        action = env.action_space.sample()  # random 0 or 1
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            print(f"Episode {episode}: score={info['score']}, steps={steps}, reward={total_reward:.1f}")
            break

env.close()