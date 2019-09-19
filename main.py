import gym

import gym
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


class OneStepActorCriticAgent:
    def __init__(self, actor_alpha, critic_alpha):
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha
        self.time_decay = 1

    def begin_episode(self):
        self.I = 1

    def policy_estimate(self, observation):
        self.

    def value(self, observation)

    def action(self, observation):
        action_distribution = self.policy_estimate(observation)


    def td_update(self, observation, next_observation, reward):
        v = self.value(observation)
        next_v = self.value(next_observation)
        delta = reward + self.time_decay*next_v - v

        self.I *= self.time_decay
