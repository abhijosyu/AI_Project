import matplotlib.pyplot as plt
import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from flappy_bird_env import FlappyBirdEnv


def moving_average(values, window=20):
    if len(values) < window:
        return values
    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        averages.append(sum(values[start:i + 1]) / (i - start + 1))
    return averages


def main():

    log_dir = "dqn_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    env = FlappyBirdEnv(render_mode=None)
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))



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

    x, y = ts2xy(load_results(log_dir), "timesteps")
    y_avg = moving_average(y, window=20)

    plt.figure()
    plt.plot(x, y, label="Episode Reward")
    plt.plot(x, y_avg, label="Moving Average Reward")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("DQN Training Performance")
    plt.legend()
    plt.tight_layout()
    title = f"dqn_buf{model.buffer_size}_learn{model.learning_starts}.png"
    plt.savefig(title, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
