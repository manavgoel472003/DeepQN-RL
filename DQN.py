import gym
from ale_py import ALEInterface, roms
from collections import deque
import torch
import torch.nn.functional as F
from torch.optim import Adam
import DQN_Agent as dqa
import random
import numpy as np

MAX_BUFFER = 50000
MIN_BUFFER = 1000
NUM_TOTAL_STEPS = 200000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = (NUM_TOTAL_STEPS * 0.95) // 1
GAMMA = 0.99


class Atari_Agent:

    def __init__(self, env: gym.Env, range_dim=None, image=False):
        self.env = env
        self.image = image
        self.range_dim = range_dim

    def training(self, weight_path="env.pth"):
        #initialising hyper_parameters
        replay_buffer = deque(maxlen=MAX_BUFFER)
        reward_buffer = deque([0.0], maxlen=100)
        target_update_freq = 500
        total_steps = 0
        episode_counter = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setting up networks
        obs, _ = self.env.reset()
        obs = dqa.pre_process(obs, self.range_dim, self.image, device=device)    
        input_dim = obs.shape if self.image else obs.shape[0]
        online_net = dqa.Agent(input_dim, num_actions=self.env.action_space.n, image_obs=self.image)
        target_net = dqa.Agent(input_dim, num_actions=self.env.action_space.n, image_obs=self.image)

        online_net.to(device)
        target_net.to(device)

        optimizer = Adam(online_net.parameters(), lr=5e-4)

        obs, _ = self.env.reset()
        obs = dqa.pre_process(obs, self.range_dim, self.image, device=device)    
        episode_reward = 0
        while total_steps < NUM_TOTAL_STEPS:
            total_steps += 1 #counter
            epsilon = np.interp(total_steps, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
            rand = torch.rand(1)
            action = online_net.act(obs) if rand >= epsilon else self.env.action_space.sample()
            new_obs, reward, done, truncated, info = self.env.step(action)
                
            new_obs = dqa.pre_process(new_obs, self.range_dim, self.image, device=device)    
            replay_buffer.append((obs.cpu(), action, reward, done, new_obs.cpu()))
            obs = new_obs
            episode_reward += reward
            if done or truncated:
                obs, _ = self.env.reset()
                obs = dqa.pre_process(obs, self.range_dim, self.image, device=device)    
                done = False
                episode_counter += 1
                reward_buffer.append(episode_reward)
                episode_reward = 0
            # set mean reward
            if len(replay_buffer) >= MIN_BUFFER:
                # do gradient descent step
                transitions = random.sample(replay_buffer, BATCH_SIZE)
                
                obses = np.asarray([t[0] for t in transitions])
                actions = np.asarray([t[1] for t in transitions])
                rewards = np.asarray([t[2] for t in transitions])
                dones = np.asarray([t[3] for t in transitions])
                new_obses = np.asarray([t[4] for t in transitions])
            
                obs_batch  = torch.tensor(obses,  dtype=torch.float32, device=device)
                action_batch  = torch.tensor(actions,dtype=torch.long, device=device)
                reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device)
                done_batch = torch.tensor(dones, dtype=torch.float32, device=device)
                new_obs_batch = torch.tensor(new_obses, dtype=torch.float32, device=device)
                
                # loss function
                with torch.no_grad():
                    max_target_q_values = target_net(new_obs_batch).max(dim=1).values
                    yi = reward_batch + (1 - done_batch) * GAMMA * max_target_q_values
                    
                online_q_values = online_net(obs_batch)
                action_q_values = online_q_values.gather(dim=1, index=action_batch.unsqueeze(-1))
                
                loss = F.mse_loss(action_q_values.squeeze(-1), yi)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % 1000 == 0 and len(reward_buffer) > 1:
                print("-"*30)
                print(f"Steps {total_steps} : Reward (Avg last 100) {np.mean(reward_buffer)}, Epsilon : {epsilon:.2f}")

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

        
        torch.save(target_net.state_dict(), weight_path)


    def play(self, n_games:int, weight_path="env.pth"):
        obs, _ = self.env.reset()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs = dqa.pre_process(obs, self.range_dim, self.image, device=device)    
        input_dim = obs.shape if self.image else obs.shape[0]
        online_net = dqa.Agent(input_dim, num_actions=self.env.action_space.n, image_obs=self.image)
        online_net.to(device=device)
        for i in range(n_games):
            obs, _ = self.env.reset()
            
            obs = dqa.pre_process(obs, self.range_dim, self.image, device=device)    
            episode_reward = 0
            done = False
            while not done:
                action = online_net.act(obs)
                new_obs, reward, done, truncated, _ = self.env.step(action)
                obs = dqa.pre_process(new_obs, self.range_dim, self.image, device=device)    
                episode_reward += reward
                done = done or truncated
            print(f"Game : {i+1} : Reward {episode_reward}")