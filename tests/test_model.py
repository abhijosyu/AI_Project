import pytest
import os
from stable_baselines3 import DQN
from flappy_bird_env import FlappyBirdEnv

def test_model_exists():
    model_path = "models/dqn_flappy.zip"
    assert os.path.exists(model_path), f"Saved model not found at {model_path}"

def test_model_loading_and_prediction():
    model_path = "models/dqn_flappy.zip"
    # Ensure it loads safely
    model = DQN.load(model_path)
    assert model is not None, "Model failed to load"
    
    env = FlappyBirdEnv(render_mode=None)
    obs, _ = env.reset()
    
    action, _states = model.predict(obs, deterministic=True)
    
    assert action in [0, 1], f"Model predicted invalid action: {action}"
