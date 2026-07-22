import numpy as np


class RandomAgent:
    def __init__(self, act_space):
        self.act_space = act_space
        self.num_actions = act_space["action"].shape[0]

    def policy(self, obs, state=None, mode="train", step=None):
        batch_size = len(next(iter(obs.values())))
        indices = np.random.randint(0, self.num_actions, size=batch_size)
        acts = np.eye(self.num_actions)[indices].astype(np.float32)
        act = {"action": acts}
        return act, state
