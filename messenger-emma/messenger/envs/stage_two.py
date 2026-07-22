# %%
"""
Classes that follows a gym-like interface and implements stage two of the Messenger
environment.
"""

import random
import re
import time
from collections import namedtuple
from os import environ
from pathlib import Path
import numpy as np
from termcolor import cprint

# hack to stop PyGame from printing to stdout
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from vgdl.interfaces.gym import VGDLEnv

import messenger.envs.config as config
from messenger.envs.base import Grid, MessengerEnv, Position
from messenger.envs.DiscreteEntity import DiscreteEntities
from messenger.envs.manual import TextManual
from messenger.envs.utils import games_from_json, text_from_path

# specifies the game variant (e.g. chasing enemy, fleeing message, stationary goal)
# path is path to the vgdl domain file describing the variant.
GameVariant = namedtuple(
    "GameVariant", ["path", "enemy_type", "message_type", "goal_type"]
)


AVATAR_KEYS = ["with_message.1", "no_message.1"]
ENEMY_TAG = "enemy.1"
MESSAGE_TAG = "message.1"
GOAL_TAG = "goal.1"


def num_entities(obs):
    # except AVATAR_KEYS
    entities = [key for key in obs.keys() if key not in AVATAR_KEYS]
    return len(entities)


def get_entity_order(obs, is_shuffle=True):
    # enemy.1, message.1, goal.1 -> order in the discrete entities
    order = list(range(num_entities(obs)))
    if is_shuffle:
        random.shuffle(order)
    res = {k: v for k, v in zip(obs.keys(), order) if k not in AVATAR_KEYS}
    assert max(res.values()) == len(res) - 1, f"{max(res.values())=}, {len(res)=}"
    return res


class StageTwo(MessengerEnv):
    """
    Full messenger environment with mobile sprites. Uses Py-VGDL as game engine.
    To avoid the need to instantiate a large number of games, (since there are
    P(12,3) = 1320 possible entity to role assignments) We apply a wrapper on top
    of the text and game state which masks the role archetypes (enemy, message goal)
    into entities (e.g. alien, knight, mage).
    """

    def __init__(
        self,
        split: str,
        is_shuffle_obs=True,
        deter_game=False,
        discrete_obs=True,

        # same_game_config_when_reset=False,
    ):
        super().__init__()
        self.split = split
        self.discrete_obs = discrete_obs
        assert split in ["train-mc", "train-sc", "val", "test", "test-se"], split

        self.deter_game = deter_game
        if deter_game:
            cprint("Deterministic game", "red")
        self.is_shuffle_obs = is_shuffle_obs  # shuffle the entity layers

        this_folder = Path(__file__).parent
        # Get the games and manual
        games_json_path = this_folder.joinpath("games.json")
        if "train" in split and "mc" in split:  # multi-combination games
            game_split = "train_multi_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")

        elif "train" in split and "sc" in split:  # single-combination games
            game_split = "train_single_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")

        elif "val" in split:
            game_split = "val"
            text_json_path = this_folder.joinpath("texts", "text_val.json")

        elif "test" in split:
            game_split = "test"
            text_json_path = this_folder.joinpath("texts", "text_test.json")

        else:
            raise Exception(f"Split: {split} not understood.")

        # list of Game namedtuples
        self.all_games = games_from_json(
            json_path=str(games_json_path), split=game_split
        )
        self.text_manual = TextManual(json_path=text_json_path)

        # get the folder that has the game variants and init_states
        if (
            "test" in split and "se" not in split
        ):  # new dynamics (se for state estimation)
            vgdl_files = this_folder.joinpath("vgdl_files", "stage_2_nd")

        else:  # training dynamics
            vgdl_files = this_folder.joinpath("vgdl_files", "stage_2")

        # get the file paths to possible starting states
        self.init_states = [
            str(path) for path in vgdl_files.joinpath("init_states").glob("*.txt")
        ]
        # get all the game variants
        self.game_variants = [
            self._get_variant(path)
            for path in vgdl_files.joinpath("variants").glob("*.txt")
        ]

        # entities tracked by VGDLEnv
        self.notable_sprites = [
            "enemy",
            "message",
            "goal",
            "no_message",
            "with_message",
        ]
        self.env = None  # the VGDLEnv

        self.only_one_game_config = None
        # if same_game_config_when_reset:
        # cprint("Use one game config for all resets", "red")
        # self.only_one_game_config = self.create_game_config()

        self.game = None
        self.variant = None
        self.init_state = None
        self.manual = None

    def _get_variant(self, variant_file: Path) -> GameVariant:
        """
        Return the GameVariant for the variant specified by variant_file.
        Searches through the vgdl code to find the correct type:
        {chaser, fleeing, immovable}
        """

        code = text_from_path(variant_file)
        return GameVariant(
            path=str(variant_file),
            enemy_type=re.search(r"enemy > (\S+)", code)[1].lower(),  # type: ignore
            message_type=re.search(r"message > (\S+)", code)[1].lower(),  # type: ignore
            goal_type=re.search(r"goal > (\S+)", code)[1].lower(),  # type: ignore
        )

    def _convert_obs(self, vgdl_obs):
        """
        Return a grid built from the vgdl observation which is a
        KeyValueObservation object (see vgdl code for details).
        """
        entity_locs = Grid(layers=3, shuffle=self.is_shuffle_obs)
        avatar_locs = Grid(layers=1)

        # try to add each entity one by one, if it's not there move on.
        # add in shuffle orders
        assert self.game is not None, "Game must be set before converting obs."
        if ENEMY_TAG in vgdl_obs:
            entity_locs.add(self.game.enemy, Position(*vgdl_obs[ENEMY_TAG]["position"]))
        if MESSAGE_TAG in vgdl_obs:
            entity_locs.add(
                self.game.message, Position(*vgdl_obs[MESSAGE_TAG]["position"])
            )
        else:
            # advance the entity counter, Oracle model requires special order.
            # TODO: maybe used named layers to make this more understandable.
            entity_locs.entity_count += 1
        if GOAL_TAG in vgdl_obs:
            entity_locs.add(self.game.goal, Position(*vgdl_obs[GOAL_TAG]["position"]))

        if "no_message.1" in vgdl_obs:
            """
            Due to a quirk in VGDL, the avatar is no_message if it starts as no_message
            even if the avatar may have acquired the message at a later point.
            To check, if it has a message, check that the class vector corresponding to
            with_message is == 1.
            """
            avatar_pos = Position(*vgdl_obs["no_message.1"]["position"])
            # with_key is last position, see self.notable_sprites
            if vgdl_obs["no_message.1"]["class"][-1] == 1:
                avatar = config.WITH_MESSAGE
            else:
                avatar = config.NO_MESSAGE

        elif "with_message.1" in vgdl_obs:
            # this case only occurs if avatar begins as with_message at start of episode
            avatar_pos = Position(*vgdl_obs["with_message.1"]["position"])
            avatar = config.WITH_MESSAGE

        else:  # the avatar is not in observation, so is probably dead
            return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

        avatar_locs.add(avatar, avatar_pos)  # if not dead, add it.

        return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

    def reset_game_config(self, **game_config):
        GAME_CONFIG_KEYS = [
            "game",
            "variant_id",
            "variant",
            "init_state_id",
            "init_state",
            "manual",
        ]
        for key in GAME_CONFIG_KEYS:
            assert key in game_config, f"{key} must be specified."
        self.only_one_game_config = game_config

    def reset(self, **kwargs):
        """
        Resets the current environment. NOTE: We remake the environment each time.
        This is a workaround to a bug in py-vgdl, where env.reset() does not
        properly reset the environment. kwargs go to get_document().
        """

        # choose the game variant (e.g. enmey-chasing, message-fleeing, goal-static)
        # and initial starting location of the entities.

        # if doesn't have game_config, create one
        # if not hasattr(self, "only_one_game_config"):
        if self.only_one_game_config is None:
            game_config = self.create_game_config()
        else:
            game_config = self.only_one_game_config

        for key, value in game_config.items():
            setattr(self, key, value)

        # args that will go into VGDL Env.
        assert self.variant is not None, "Variant must be specified."
        self._envargs = {
            "game_file": self.variant.path,
            "level_file": self.init_state,
            "notable_sprites": self.notable_sprites.copy(),
            "obs_type": "objects",  # track the objects
            "block_size": 34,  # rendering block size
            "deter_game": self.deter_game,
        }
        self.env = VGDLEnv(**self._envargs)
        vgdl_obs = self.env.reset()
        # dict[str in ['enemy.1', 'message.1', 'goal.1'], order in the discrete entities]
        # e.g. {'enemy.1': 0, 'message.1': 1, 'goal.1': 2}

        self.role2order = get_entity_order(vgdl_obs)
        self.order2role = {v: k for k, v in self.role2order.items()}
        self.role_order_list = [self.order2role[i] for i in range(len(self.role2order))]

        manual_order = [0, 1, 2]  # order of manuals in the text
        if self.is_shuffle_obs:
            random.shuffle(manual_order)

        assert self.manual is not None, "Manual must be specified."
        # print(f"non-shuffle {self.manual=}")
        self.shuffled_manual = [self.manual[i] for i in manual_order]

        # original manual [0,1,2] -> [enemy, message, goal]
        ORIGIN_MANUAL_ID_2_ROLE = {
            0: ENEMY_TAG,
            1: MESSAGE_TAG,
            2: GOAL_TAG,
        }
        # role 2 manual id in the shuffle manual
        # e.g. shuffle manual is [goal, enemy, message] -> [2, 0, 1]
        # -> role2manual_id in this shuffle manual = ['goal': 0, 'enemy': 1, 'message': 2]
        self.role_2_shuffle_manual_id = {
            role: manual_order.index(origin_manual_id)
            for origin_manual_id, role in ORIGIN_MANUAL_ID_2_ROLE.items()
        }
        self.manual_ids = [
            self.role_2_shuffle_manual_id[k] for k in self.role_order_list
        ]
        # turn to np array
        self.manual_ids = np.array(self.manual_ids)

        role2movement = {
            ENEMY_TAG: self.variant.enemy_type,
            MESSAGE_TAG: self.variant.message_type,
            GOAL_TAG: self.variant.goal_type,
        }
        self.movement_classes = [role2movement[k] for k in self.role_order_list]

        # return self._convert_obs(vgdl_obs), shuffled_manual
        # self.vgdl_obs = vgdl_obs
        # if not test_sent_emb_only:
        #     self.test_sent_emb_only = test_sent_emb_only
        #     obs = {
        #         **self._convert_obs_to_discrete(vgdl_obs),
        #         **self._convert_obs(vgdl_obs),
        #     }
        # else:

        if self.discrete_obs:
            obs = self._convert_obs_to_discrete(vgdl_obs)
        else:
            obs = self._convert_obs(vgdl_obs)

        return obs, self.shuffled_manual

    def create_game_config(self):
        if self.only_one_game_config is not None:
            cprint("Create game config", "yellow")

        random.seed(time.time())
        game_id = random.choice(range(len(self.all_games)))
        game = self.all_games[game_id]  # for manual and dynamics
        variant_id = random.choice(range(len(self.game_variants)))
        variant = self.game_variants[variant_id]
        init_state_id = random.choice(range(len(self.init_states)))
        init_state = self.init_states[init_state_id]
        manual = self.text_manual.get_document(
            enemy=game.enemy.name,
            message=game.message.name,
            goal=game.goal.name,
            enemy_type=variant.enemy_type,
            message_type=variant.message_type,
            goal_type=variant.goal_type,
        )

        return {
            "game": game,
            "variant_id": variant_id,
            "variant": variant,
            "init_state_id": init_state_id,
            "init_state": init_state,
            "manual": manual,
        }

    def step(self, action):
        assert self.env is not None, "Must call reset() before step()."
        vgdl_obs, reward, done, info = self.env.step(action)
        # self.vgdl_obs = vgdl_obs
        # return self._convert_obs(vgdl_obs), reward, done, info
        # if hasattr(self, "test_sent_emb_only"):
        #     if not self.test_sent_emb_only:
        #         obs = {
        #             **self._convert_obs_to_discrete(vgdl_obs),
        #             **self._convert_obs(vgdl_obs),
        #         }
        # else:
        if self.discrete_obs:
            obs = self._convert_obs_to_discrete(vgdl_obs)
        else:
            obs = self._convert_obs(vgdl_obs)
        return obs, reward, done, info

    def _convert_obs_to_discrete(self, vgdl_obs):
        disc_entities = DiscreteEntities(self.role2order)
        disc_avatar = DiscreteEntities(is_avatar=True)
        assert self.game is not None, "Game must be set before converting obs."
        # print("movement_classes", self.movement_classes)
        if ENEMY_TAG in vgdl_obs:
            disc_entities.add(
                self.game.enemy,
                Position(*vgdl_obs[ENEMY_TAG]["position"]),
                # self.entity2manual["enemy.1"],
                self.role_2_shuffle_manual_id[ENEMY_TAG],
                ENEMY_TAG,
            )

        if MESSAGE_TAG in vgdl_obs:
            disc_entities.add(
                self.game.message,
                Position(*vgdl_obs[MESSAGE_TAG]["position"]),
                # self.entity2manual["message.1"],
                self.role_2_shuffle_manual_id[MESSAGE_TAG],
                MESSAGE_TAG,
            )

        if GOAL_TAG in vgdl_obs:
            disc_entities.add(
                self.game.goal,
                Position(*vgdl_obs[GOAL_TAG]["position"]),
                # self.entity2manual["goal.1"],
                self.role_2_shuffle_manual_id[GOAL_TAG],
                GOAL_TAG,
            )

        if "no_message.1" in vgdl_obs:
            """
            Due to a quirk in VGDL, the avatar is no_message if it starts as no_message
            even if the avatar may have acquired the message at a later point.
            To check, if it has a message, check that the class vector corresponding to
            with_message is == 1.
            """
            avatar_pos = Position(*vgdl_obs["no_message.1"]["position"])
            # with_key is last position, see self.notable_sprites
            if vgdl_obs["no_message.1"]["class"][-1] == 1:
                avatar = config.WITH_MESSAGE
            else:
                avatar = config.NO_MESSAGE

        elif "with_message.1" in vgdl_obs:
            # this case only occurs if avatar begins as with_message at start of episode
            avatar_pos = Position(*vgdl_obs["with_message.1"]["position"])
            avatar = config.WITH_MESSAGE
        else:
            # the avatar is not in observation, so is probably dead
            # assert disc_entities.manual_ids and sel.shuffled_manual_ids are the same, same np array
            # assert (disc_entities.manual_ids == self.manual_ids).all()
            # ignore when disc_entitiets.manual_ids == -1
            is_alive = disc_entities.manual_ids != -1
            manual_ids_alive = disc_entities.manual_ids[is_alive]
            assert (manual_ids_alive == self.manual_ids[is_alive]).all()

            return {
                "entity_ids": disc_entities.ids,
                "entity_pos": disc_entities.pos,
                "manual_ids": disc_entities.manual_ids,
                "avatar_ids": disc_avatar.ids,
                "avatar_pos": disc_avatar.pos,
            }

        disc_avatar.add(avatar, avatar_pos)
        return {
            "entity_ids": disc_entities.ids,
            "entity_pos": disc_entities.pos,
            "manual_ids": disc_entities.manual_ids,
            "avatar_ids": disc_avatar.ids,
            "avatar_pos": disc_avatar.pos,
        }


# test stage_two
def test_env():
    env = StageTwo("train-mc")
    obs, manual = env.reset()
    print(obs)
    print(manual)


if __name__ == "__main__":
    test_env()
