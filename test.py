from flappy_bird_env import FlappyBirdEnv
from gymnasium.utils.env_checker import check_env

env = FlappyBirdEnv(render_mode="human")

check_env(env)
print("Environment passed all checks!")

obs, _ = env.reset()


for _ in range(60):
    obs, reward, done, _, info = env.step(0)  # do nothing, let bird fall
    print(f"bird_y={obs[0]:.3f}, speed={obs[1]:.3f}, pipe_dist={obs[2]:.3f}, gap_top={obs[3]:.3f}, gap_bot={obs[4]:.3f}")
    if done:
        break



env.close()