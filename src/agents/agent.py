class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def begin_episode(self):
        """ Initialization of agent at beginning of an episode """
        raise NotImplementedError()

    def step(self, observation, reward):
        """ At a new step of an episode, return an action """
        raise NotImplementedError()

    def end_episode(self):
        """ Call at the end of an episode. """
        raise NotImplementedError()
