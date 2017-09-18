import gym
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_env(env_id, **kwargs):
    return create_atari_env(env_id)

def create_atari_env(env_id):
    env = gym.make(env_id)
    return env
