from guppy import hpy
import gym
import numpy as np
from .utils import get_neuron_number_from_space
from .agents.dqn_agent import DQNAgent

h = hpy()

env = gym.make('CartPole-v0')

agent = DQNAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    epsilon=0.2,
    step_size=0.01,
    time_decay=0.9
)

for i_episode in range(10):
    observation = env.reset()
    action = agent.begin_episode(observation)
    for t in range(100):
        # env.render()
        observation, reward, done, info = env.step(action)
        action = agent.step(next_observation=observation, reward=reward, done=done)
        if done:
            agent.end_episode()
            break
    print("Episode finished after {} timesteps".format(t+1))



def performance_measure(agent):
    # Performance Measure 1: Number of timesteps.
    # Performance Measure 2: Intuitively, the agent should move left if the angle is negative, and the agent should move right if the angle is positive.

    observation = env.reset()
    action = agent.begin_episode(observation)
    measure_two = 0
    for t in range(100):
        observation, reward, done, info = env.step(action)
        action = agent.step(next_observation=observation, reward=reward, done=done)
        # np.sign(observation[2]) is -1 if angle is to left, +1 if angle is to right
        # (-1 + 2*action) is -1 if moving left, and +1 if moving right
        measure_two += np.sign(observation[2]) * (-1 + 2*action)
        if done:
            agent.end_episode()
            break
    measure_two /= (t+1)
    print("Measure 1: {}/100".format(t+1))
    print("Measure 2: {}".format(measure_two))  

print(h.heap())

performance_measure(agent)
env.close()


