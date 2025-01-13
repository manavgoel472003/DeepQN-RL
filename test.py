import gym
from ale_py import ALEInterface, roms
import DQN as dqn

# Test - 1

env = gym.make('Acrobot-v1', render_mode="human")

cartpole_agent = dqn.Atari_Agent(env)
path = "Acrobot.pth"
# cartpole_agent.training(weight_path= path)
cartpole_agent.play(n_games=5, weight_path= path)

# Test - 2
# ale = ALEInterface()
# ale.loadROM(roms.Pong)
# ale.reset_game()
# env = gym.make("ALE/Pong", render_mode="human")

# pong_agent = dqn.Atari_Agent(env, range_dim=(35, 195), image=True)
# path = "pong.pth"
# # pong_agent.training(weight_path=path)
# pong_agent.play(n_games=5, weight_path=path)