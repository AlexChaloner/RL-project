import numpy as np
from .agent import Agent


class OnPolicyFirstVisitMonteCarloAgent(Agent):
    """ From page 101 of Reinforcement Learning: An Introduction (2nd Edition) by Sutton and Barto """
    def __init__(self, action_space, observation_space, epsilon, time_decay):
        super().__init__(action_space)
        self.epsilon = epsilon
        self.time_decay = time_decay
        self._initialise_policy(observation_space, epsilon)
        self.policy = np.zeros((observation_space.size, action_space.size))
        self.returns = np.full((observation_space.size, action_space.size), [])
        self.Q = np.zeros((observation_space.size, action_space.size))

    def _initialise_policy(self, observation_space, epsilon):
        if epsilon > 1 or epsilon < 0:
            raise ValueError('Epsilon should be between 0 and 1')
        for observation in observation_space:
            # epsilon-soft policy
            for action in self.action_space:
                self.policy[observation, action] = epsilon/self.action_space.size
            if epsilon < 1:
                action = self.action_space.sample()
                self.policy[observation, action] += 1 - epsilon

    def _choose_action(self, observation):
        distribution = self.policy[observation]
        return np.random.choice(distribution)

    def begin_episode(self):
        self.episode_history = []

    def step(self, observation, reward):
        action = self._choose_action(observation)
        self.episode_history.append((observation, action, reward))
        return action

    def end_episode(self):
        G = 0
        T = len(self.episode_history)
        # Loop for each step of episode, t = T-1, T-2, ..., 0
        for t in range(T-1, -1, -1):
            observation, action, reward = self.episode_history[t]
            G = self.time_decay*G + reward
            if (observation, action) not in [(obs, act) for obs, act, _ in self.episode_history[0:t]]:
                self.returns[observation, action].append(G)
                self.Q[observation, action] = np.mean(self.returns[observation, action])
                optimal_action = np.argmax(self.Q[observation])
                for action in self.action_space:
                    if action == optimal_action:
                        self.policy[observation, action] = 1 - self.epsilon + self.epsilon/self.action_space.size
                    else:
                        self.policy[observation, action] = self.epsilon/self.action_space.size
