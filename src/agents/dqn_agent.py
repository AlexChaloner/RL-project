import numpy as np
from .agent import Agent
from tensorflow import keras
# pylint: disable=import-error
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Input, Flatten, Reshape, Conv2D, MaxPool2D, Dense, Activation
# pylint: enable=import-error
import random
from ..utils import get_neuron_number_from_space
from functools import reduce



@profile
class ReplayBuffer:
    def __init__(self):
        self.replay = []
        self._max_length = 1000

    def add(self, observation, action, reward, next_observation, done):
        step = (
            observation,
            action,
            reward,
            next_observation,
            done
        )
        self.replay.append(step)
        # Efficiently cut down list size
        if len(self.replay) > self._max_length * 1.1:
            print('Cutting down Replay Buffer')
            self.replay = self.replay[-self._max_length:]

    def clear(self):
        self.replay = []

    def get_batch(self, batch_size=32):
        if len(self.replay) < batch_size:
            batch_size = len(self.replay)
        return random.sample(self.replay, batch_size)


@profile
class DQNAgent(Agent):
    def __init__(self, action_space, observation_space, epsilon, step_size, time_decay):
        super().__init__(action_space)
        self.epsilon = epsilon
        self.step_size = step_size
        self.time_decay = time_decay
        self._initialise_policy(observation_space, action_space, epsilon)
        self.replay = ReplayBuffer()

    def _initialise_policy(self, observation_space, action_space, epsilon):
        if epsilon > 1 or epsilon < 0:
            raise ValueError('Epsilon should be between 0 and 1')
        self._build_q_network(observation_space, action_space)

    def _build_q_network(self, observation_space, action_space):
        self.Q = keras.models.Sequential()
        input_shape = get_neuron_number_from_space(observation_space)
        if isinstance(input_shape, tuple):
            self.Q.add(Input(shape=input_shape))
            self.Q.add(Flatten())
        elif isinstance(input_shape, int):
            self.Q.add(Input(shape=input_shape))
        else:
            raise NotImplementedError('Input shape {} is not implemented'.format(input_shape))
        self.Q.add(Dense(4, activation='relu'))
        output_shape = get_neuron_number_from_space(action_space)
        if isinstance(output_shape, tuple):
            output_number = reduce((lambda x, y: x * y), list(output_shape))
            self.Q.add(Dense(output_number))
            self.Q.add(Reshape(output_shape))
        elif isinstance(output_shape, int):
            self.Q.add(Dense(output_shape))
        else:
            raise NotImplementedError('Output shape {} is not implemented'.format(output_shape))
        self.Q.compile(optimizer='adam', loss=Huber())

    def _choose_action(self, observation):
        # Epsilon-greedy DQN
        if np.random.rand() > self.epsilon:
            action_values = self.Q.predict(np.array([observation]))
            action = np.argmax(action_values)
        else:
            action = self.action_space.sample()
        return action

    def begin_episode(self, observation):
        self.observation = observation
        self.action = self._choose_action(observation)
        return self.action

    def step(self, next_observation, reward, done):
        self.replay.add(self.observation, self.action, reward, next_observation, done)
        self.observation = next_observation
        self.action = self._choose_action(self.observation)
        return self.action

    def end_episode(self):
        self.train()

    def train(self):
        transition_batch = self.replay.get_batch(batch_size=32)
        x_batch = []
        y_batch = []
        for observation, action, reward, next_observation, done in transition_batch:
            if done:
                aim_value = reward
            else:
                action_values = self.Q.predict(np.array([next_observation]))[0]
                max_Q = max(action_values)
                aim_value = reward + self.time_decay*max_Q
            current_Q = self.Q.predict(np.array([observation]))[0]
            x_batch.append(observation)
            y = current_Q
            y[action] = aim_value
            # print('Aim:', y)
            y_batch.append(y)
        self.Q.train_on_batch(x=np.array(x_batch), y=np.array(y_batch))
