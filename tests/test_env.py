import pytest
import numpy as np
from flappy_bird_env import FlappyBirdEnv

def test_env_initialization():
    env = FlappyBirdEnv(render_mode=None)
    assert env.action_space.n == 2, "Action space should be discrete 2 (flap/do nothing)"
    assert env.observation_space.shape == (5,), "Observation space should have 5 features"

def test_env_reset():
    env = FlappyBirdEnv(render_mode=None)
    obs, info = env.reset()
    assert obs.shape == (5,), "Reset should return properly shaped initial observation"
    assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
    assert isinstance(info, dict), "Info from reset should be a dictionary"
    assert env.score == 0, "Initial score should be 0"

def test_env_step():
    env = FlappyBirdEnv(render_mode=None)
    env.reset()
    
    # Take a 'flap' action
    obs, reward, terminated, truncated, info = env.step(1)
    
    assert obs.shape == (5,), "Observation from step is malformed"
    assert isinstance(reward, float), "Reward should be a float value"
    assert isinstance(terminated, bool), "Terminal flag missing or wrong type"
    assert isinstance(truncated, bool), "Truncated flag missing or wrong type"
    assert "score" in info, "Score info missing from step return"
