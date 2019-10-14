import numpy as np
import gym


def get_neuron_number_from_space(space):
    # Utility function for creating neuron layers given a space; made by inspecting https://github.com/openai/gym/tree/master/gym/spaces
    if isinstance(space, gym.spaces.Box):
        # Box has a high, low and shape value.
        # Box.shape details the dimensions of the space. So there should be neurons in this shape.
        # High and low can be ignored
        return space.shape
    elif isinstance(space, gym.spaces.Discrete):
        # space.n is the discrete number of items in the space. Hence there should be space.n neurons
        return space.n
    elif isinstance(space, gym.spaces.MultiBinary):
        # space.n is the discrete number of binary values in the space.
        return space.n
    elif isinstance(space, gym.spaces.MultiDiscrete):
        # MultiDiscrete has a vector of n's, being 
        print(space.nvec)
        print(np.sum(space.nvec))
        return np.sum(space.nvec)
    elif isinstance(space, gym.spaces.Tuple):
        print(space.spaces)
        neurons = []
        for subspace in space.spaces:
            neurons.append(get_neuron_number_from_space(subspace))
        return neurons
    else:
        raise NotImplementedError('Could not recognise space class {}'.format(space.__class__))
