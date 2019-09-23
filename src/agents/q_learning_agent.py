import numpy as np
from .agent import Agent


class QLearningAgent(Agent):
    """ From page 101 of Reinforcement Learning: An Introduction (2nd Edition) by Sutton and Barto """
    def __init__(self, action_space, observation_space, epsilon, step_size, time_decay):
        super().__init__(action_space)
        self.epsilon = epsilon
        self.step_size = step_size
        self.time_decay = time_decay
        self._initialise_policy(observation_space, epsilon)
        self.policy = np.zeros((observation_space.size, action_space.size))
        self.Q = np.zeros((observation_space.size, action_space.size))

    def _initialise_policy(self, observation_space, epsilon):
        if epsilon > 1 or epsilon < 0:
            raise ValueError('Epsilon should be between 0 and 1')
        for observation in observation_space:
            for action in self.action_space:
                self.policy[observation, action] = epsilon/self.action_space.size
            if epsilon < 1:
                action = self.action_space.sample()
                self.policy[observation, action] += 1 - epsilon

    def _choose_action(self, observation):
        # Epsilon-greedy policy
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.policy[observation])
        else:
            action = self.action_space.sample()
        return action

    def begin_episode(self, observation):
        self.observation = observation
        self.action = self._choose_action(observation)
        return self.action

    def step(self, next_observation, reward):
        self.Q[self.observation, self.action] += self.step_size*(reward + self.time_decay*np.max(self.Q[next_observation]) - self.Q[self.observation, self.action])
        self.action = self._choose_action(next_observation)
        self.observation = next_observation
        return self.action
