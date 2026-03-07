from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from flappy_bird_env import FlappyBirdEnv

def main():
    env = FlappyBirdEnv(render_mode=None)
    env = Monitor(env)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=64,
        train_freq=4,
        target_update_interval=1_000,
    )

    model.learn(total_timesteps=500_000)

    model.save("dqn_flappy")

    env.close()

if __name__ == "__main__":
    main()
