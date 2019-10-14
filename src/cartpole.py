import gym
import numpy as np
from .utils import get_neuron_number_from_space
from .agents.dqn_agent import DQNAgent

env = gym.make('CartPole-v0')

agent = DQNAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    epsilon=0.1,
    step_size=0.1,
    time_decay=0.99
)

for i_episode in range(20):
    observation = env.reset()
    action = agent.begin_episode(observation)
    for t in range(100):
        env.render()
        observation, reward, done, info = env.step(action)
        action = agent.step(next_observation=observation, reward=reward, done=done)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
