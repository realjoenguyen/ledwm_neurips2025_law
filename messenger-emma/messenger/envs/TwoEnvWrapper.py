"""
Implements wrappers on top of the basic messenger environments
"""

from copy import deepcopy
import random
from messenger.envs.base import MessengerEnv


class TwoEnvWrapper(MessengerEnv):
    """
    Switches between two Messenger environments
    """

    def __init__(
        self,
        stage: int,
        split_1: str,
        split_2: str,
        prob_env_1=0.5,
        **kwargs,
    ):
        super().__init__()
        if stage == 1:
            from messenger.envs.stage_one import StageOne

            self.env_1 = StageOne(split=split_1, **kwargs)
            self.env_2 = StageOne(split=split_2, **kwargs)
        elif stage == 2:
            from messenger.envs.stage_two import StageTwo

            self.env_1 = StageTwo(split=split_1, **kwargs)
            self.env_2 = StageTwo(split=split_2, **kwargs)
        elif stage == 3:
            from messenger.envs.stage_three import StageThree

            self.env_1 = StageThree(split=split_1, **kwargs)
            self.env_2 = StageThree(split=split_2, **kwargs)
        else:
            raise ValueError(f"Unknown Messenger stage: {stage}")

        self.prob_env_1 = prob_env_1
        self.cur_env = None
        self.only_one_game_config = None

    def get_cur_env(self) -> "StageOne|StageTwo|StageThree":
        random.seed()
        if random.random() < self.prob_env_1:
            return self.env_1
        else:
            return self.env_2

    def reset(self):
        if self.only_one_game_config is None:
            self.cur_env = self.get_cur_env()
        else:
            self.cur_env = deepcopy(self.only_one_game_config["cur_env"])
            cur_env_config = {
                k: v for k, v in self.only_one_game_config.items() if k != "cur_env"
            }
            self.cur_env.reset_game_config(**cur_env_config)
        return self.cur_env.reset()

    def step(self, action):
        assert self.cur_env is not None, "Must call reset() before step()"
        return self.cur_env.step(action)

    # @property
    # def game_id(self):
    #     assert self.cur_env is not None, "Must call reset() before game_id"
    #     return self.cur_env.game_id

    def reset_game_config(self, **kwargs):
        self.only_one_game_config = kwargs

    def create_game_config(self):
        cur_env = self.get_cur_env()
        game_config = cur_env.create_game_config()
        game_config["cur_env"] = cur_env
        return game_config

    # # attribute is called to self.cur_env
    # def __getattr__(self, name):
    #     if name.startswith("__"):
    #         raise AttributeError(name)
    #     try:
    #         return getattr(self.cur_env, name)
    #     except AttributeError:
    #         raise ValueError(name)
