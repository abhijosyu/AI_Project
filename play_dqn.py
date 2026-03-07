from stable_baselines3 import DQN
from flappy_bird_env import FlappyBirdEnv

def main():
    env = FlappyBirdEnv(render_mode="human")
    model = DQN.load("dqn_flappy")

    obs, info = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished. Score:", info["score"])
            obs, info = env.reset()

if __name__ == "__main__":
    main()
