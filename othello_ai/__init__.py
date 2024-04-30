from othello_ai.othello import OthelloEnv
from gymnasium.envs.registration import register

register(
     id="othello_ai/Othello-v0",
     entry_point="othello_ai:OthelloEnv",
     max_episode_steps=300,
)
