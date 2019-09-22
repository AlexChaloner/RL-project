import gym
import np

env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def begin_episode(self):
        """ Initialization of agent at beginning of an episode """
        raise NotImplementedError()

    def step(self, observation):
        """ At a new step of an episode, return an action """
        raise NotImplementedError()

    def end_episode(self):
        """ Call at the end of an episode. """
        raise NotImplementedError()
    

class OnPolicyFirstVisitMonteCarloAgent(Agent):
    """ From page 101 of Reinforcement Learning: An Introduction (2nd Edition) by Sutton and Barto """
    def __init__(self, action_space, state_space, epsilon, time_decay):
        super().__init__(action_space)
        self.epsilon = epsilon
        self.time_decay = time_decay
        self._initialise_policy(state_space, epsilon)
        self.policy = np.zeros((state_space.size, action_space.size))
        self.returns = np.full((state_space.size, action_space.size), [])
        self.Q = np.zeros((state_space.size, action_space.size))

    def _initialise_policy(self, state_space, epsilon):
        for state in state_space:
            for action in self.action_space:
                self.policy[state, action] = epsilon/self.action_space.size
            if epsilon < 1:
                action = self.action_space.sample()
                self.policy[state, action] += 1 - epsilon

    def _choose_action(self, observation):
        distribution = self.policy(observation)
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
                        self.policy(observation)[action] = 1 - self.epsilon + self.epsilon/len(self.action_space)
                    else:
                        self.policy(observation)[action] = self.epsilon/len(self.action_space)
