"""
Classes that follows a gym-like interface and implements stage three of the Messenger
environment.
"""

import random
from collections import namedtuple
from pathlib import Path
from os import environ
import re

# hack to stop PyGame from printing to stdout
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from messenger.envs.DiscreteEntity import DiscreteEntities
from messenger.envs.stage_two import get_entity_order
import numpy as np
from vgdl.interfaces.gym import VGDLEnv

from messenger.envs.base import MessengerEnv, Grid, Position
import messenger.envs.config as config
from messenger.envs.manual import TextManual, Descr
from messenger.envs.utils import games_from_json, text_from_path


# specifies the game variant path is path to the vgdl domain file describing the variant.
GameVariant = namedtuple(
    "GameVariant",
    [
        "path",
        "enemy_type",
        "message_type",
        "goal_type",
        "decoy_message_type",
        "decoy_goal_type",
    ],
)

ENEMY_TAG = "enemy.1"
MESSAGE_TAG = "message.1"
GOAL_TAG = "goal.1"
DECOY_MESSAGE_TAG = "decoy_message.1"
DECOY_GOAL_TAG = "decoy_goal.1"
DECOY_OTHER = "decoy_other"
ORIGIN_MANUAL_ID_2_ROLE = {
    0: ENEMY_TAG,
    1: MESSAGE_TAG,
    2: GOAL_TAG,
    3: DECOY_MESSAGE_TAG,
    4: DECOY_GOAL_TAG,
    5: DECOY_OTHER,
}


class StageThree(MessengerEnv):
    """
    Similar to stage two Messenger, except with decoy objects that require
    disambiduation (e.g. chasing knight, vs immovable knight)
    """

    def __init__(
        self,
        split: str,
        is_shuffle_obs=True,
        # same_game_config_when_reset=False,
    ):
        super().__init__()
        self.shuffle_obs = is_shuffle_obs  # shuffle the entity layers

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

        self.split = split
        # list of Game namedtuples
        self.all_games = games_from_json(json_path=games_json_path, split=game_split)
        self.text_manual = TextManual(json_path=text_json_path)

        vgdl_files = this_folder.joinpath("vgdl_files", "stage_3")

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
            "decoy_message",
            "decoy_goal",
            "no_message",
            "with_message",
        ]
        self.env = None  # the VGDLEnv
        self.only_one_game_config = None

    def reset_game_config(self, **game_config):
        GAME_CONFIG_KEYS = [
            "game",
            # "variant_id",
            "variant",
            # "init_state_id",
            "init_state",
            "manual",
        ]
        for key in GAME_CONFIG_KEYS:
            assert key in game_config, f"{key} must be specified."
        self.only_one_game_config = game_config

    def create_game_config(self):
        game_id = random.choice(range(len(self.all_games)))
        game = self.all_games[game_id]
        # choose the game variant (e.g. enmey-chasing, message-fleeing, goal-static)
        # and initial starting location of the entities.
        # if variant_id is not None:
        # variant = self.game_variants[variant_id]
        # else:
        variant = random.choice(self.game_variants)
        init_state = random.choice(self.init_states)  # inital state file

        all_npcs = (
            Descr(
                entity=game.enemy.name,
                role="enemy",
                type=variant.enemy_type,
            ),
            Descr(
                entity=game.message.name,
                role="message",
                type=variant.message_type,
            ),
            Descr(
                entity=game.goal.name,
                role="goal",
                type=variant.goal_type,
            ),
            # if touch these decoy then die,
            Descr(
                entity=game.message.name,
                role="enemy",
                type=variant.decoy_message_type,
            ),
            Descr(
                entity=game.goal.name,
                role="enemy",
                type=variant.decoy_goal_type,
            ),
        )

        manual = self.text_manual.get_document_plus(*all_npcs)
        manual.append(
            self.text_manual.get_decoy_descriptor(
                entity=game.enemy.name,
                not_of_role="enemy",
                not_of_type=variant.enemy_type,
            )
        )

        return {
            "game": game,
            "variant": variant,
            "init_state": init_state,
            "manual": manual,
        }

    def _get_variant(self, variant_file: Path) -> GameVariant:
        """
        Return the GameVariant for the variant specified by variant_file.
        Searches through the vgdl code to find the correct type:
        {chaser, fleeing, immovable}
        """

        code = text_from_path(variant_file)
        return GameVariant(
            path=str(variant_file),
            enemy_type=re.search(r"enemy > (\S+)", code)[1].lower(),
            message_type=re.search(r"message > (\S+)", code)[1].lower(),
            goal_type=re.search(r"goal > (\S+)", code)[1].lower(),
            decoy_message_type=re.search(r"decoy_message > (\S+)", code)[1].lower(),
            decoy_goal_type=re.search(r"decoy_goal > (\S+)", code)[1].lower(),
        )

    def _convert_obs(self, vgdl_obs, entity_key2manual_id=None):
        """
        Return a grid built from the vgdl observation which is a
        KeyValueObservation object (see vgdl code for details).
        """
        entity_locs = Grid(
            layers=5,
            shuffle=self.shuffle_obs,
            record_manual_ids=entity_key2manual_id is not None,
        )
        avatar_locs = Grid(layers=1)

        # try to add each entity one by one, if it's not there move on.
        if ENEMY_TAG in vgdl_obs:
            entity_locs.add(
                self.game.enemy,
                Position(*vgdl_obs[ENEMY_TAG]["position"]),
                manual_id=(
                    entity_key2manual_id[ENEMY_TAG] if entity_key2manual_id else None
                ),
            )

        if MESSAGE_TAG in vgdl_obs:
            entity_locs.add(
                self.game.message,
                Position(*vgdl_obs[MESSAGE_TAG]["position"]),
                manual_id=(
                    entity_key2manual_id[MESSAGE_TAG] if entity_key2manual_id else None
                ),
            )
        else:
            # advance the entity counter, Oracle model requires special order.
            # TODO: maybe used named layers to make this more understandable.
            entity_locs.entity_count += 1

        if GOAL_TAG in vgdl_obs:
            entity_locs.add(
                self.game.goal,
                Position(*vgdl_obs[GOAL_TAG]["position"]),
                manual_id=(
                    entity_key2manual_id[GOAL_TAG] if entity_key2manual_id else None
                ),
            )

        if DECOY_MESSAGE_TAG in vgdl_obs:
            entity_locs.add(
                self.game.message,
                Position(*vgdl_obs[DECOY_MESSAGE_TAG]["position"]),
                manual_id=(
                    entity_key2manual_id[DECOY_MESSAGE_TAG]
                    if entity_key2manual_id
                    else None
                ),
            )
        if DECOY_GOAL_TAG in vgdl_obs:
            entity_locs.add(
                self.game.goal,
                Position(*vgdl_obs[DECOY_GOAL_TAG]["position"]),
                manual_id=(
                    entity_key2manual_id[DECOY_GOAL_TAG]
                    if entity_key2manual_id
                    else None
                ),
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

        else:  # the avatar is not in observation, so is probably dead
            res = {"entities": entity_locs.grid, "avatar": avatar_locs.grid}
            if hasattr(entity_locs, "manual_ids"):
                res.update({"manual_ids": self.update_manual_ids(entity_locs)})
            return res

        avatar_locs.add(avatar, avatar_pos)  # if not dead, add it.
        res: dict = {"entities": entity_locs.grid, "avatar": avatar_locs.grid}
        if hasattr(entity_locs, "manual_ids"):
            res.update({"manual_ids": self.update_manual_ids(entity_locs)})
        return res

    def update_manual_ids(self, entity_locs):
        res = entity_locs.manual_ids[np.where(entity_locs.manual_ids >= 0)]
        # assert res.shape[0] == 5, res.shape
        if res.shape[0] < 5:
            # add -1
            res = np.concatenate([res, -1 * np.ones(5 - res.shape[0], dtype=int)])

        return res

    def reset(self, **kwargs):
        """
        Resets the current environment. NOTE: We remake the environment each time.
        This is a workaround to a bug in py-vgdl, where env.reset() does not
        properly reset the environment. kwargs go to get_document().
        """

        if self.only_one_game_config is None:
            game_config = self.create_game_config()
        else:
            game_config = self.only_one_game_config

        for key, value in game_config.items():
            setattr(self, key, value)

        # args that will go into VGDL Env.
        self._envargs = {
            "game_file": self.variant.path,
            "level_file": self.init_state,
            "notable_sprites": self.notable_sprites.copy(),
            "obs_type": "objects",  # track the objects
            "block_size": 34,  # rendering block size
        }
        self.env = VGDLEnv(**self._envargs)
        vgdl_obs = self.env.reset()

        self.role2order = get_entity_order(vgdl_obs)
        self.order2role = {v: k for k, v in self.role2order.items()}
        self.role_order_list = [self.order2role[i] for i in range(len(self.role2order))]

        # enemy, message, goal, enemy, enemy
        manual_order = list(range(len(self.manual)))
        if self.shuffle_obs:
            random.shuffle(manual_order)
        self.shuffled_manual = [self.manual[i] for i in manual_order]

        # entity_keys = [
        #     ENEMY_TAG,
        #     MESSAGE_TAG,
        #     GOAL_TAG,
        #     DECOY_MESSAGE_TAG,
        #     DECOY_GOAL_TAG,
        # ]

        self.role_2_shuffle_manual_id = {
            role: manual_order.index(origin_manual_id)
            for origin_manual_id, role in ORIGIN_MANUAL_ID_2_ROLE.items()
        }
        self.manual_ids = [
            self.role_2_shuffle_manual_id[k] for k in self.role_order_list
        ]
        self.manual_ids = np.array(self.manual_ids)

        role2movement = {
            ENEMY_TAG: self.variant.enemy_type,
            MESSAGE_TAG: self.variant.message_type,
            GOAL_TAG: self.variant.goal_type,
            DECOY_MESSAGE_TAG: self.variant.decoy_message_type,
            DECOY_GOAL_TAG: self.variant.decoy_goal_type,
        }
        self.movement_classes = [role2movement[k] for k in self.role_order_list]

        # return self._convert_obs(vgdl_obs, self.entity_key2manual_id), shuffled_manual
        # self.vgdl_obs = vgdl_obs

        disc_obs = self._convert_obs_to_discrete(vgdl_obs)
        return disc_obs, self.shuffled_manual

    def step(self, action):
        assert self.env is not None, "Must call reset() before step()."
        vgdl_obs, reward, done, info = self.env.step(action)
        # self.vgdl_obs = vgdl_obs
        # return (
        #     self._convert_obs(vgdl_obs, self.entity_key2manual_id),
        #     reward,
        #     done,
        #     info,
        # )
        return self._convert_obs_to_discrete(vgdl_obs), reward, done, info

    def _convert_obs_to_discrete(self, vgdl_obs):
        disc_entities = DiscreteEntities(self.role2order)
        disc_avatar = DiscreteEntities(is_avatar=True)

        if ENEMY_TAG in vgdl_obs:
            disc_entities.add(
                self.game.enemy,
                Position(*vgdl_obs[ENEMY_TAG]["position"]),
                self.role_2_shuffle_manual_id[ENEMY_TAG],
                ENEMY_TAG,
            )
        if MESSAGE_TAG in vgdl_obs:
            disc_entities.add(
                self.game.message,
                Position(*vgdl_obs[MESSAGE_TAG]["position"]),
                self.role_2_shuffle_manual_id[MESSAGE_TAG],
                MESSAGE_TAG,
            )
        if GOAL_TAG in vgdl_obs:
            disc_entities.add(
                self.game.goal,
                Position(*vgdl_obs[GOAL_TAG]["position"]),
                self.role_2_shuffle_manual_id[GOAL_TAG],
                GOAL_TAG,
            )
        if DECOY_MESSAGE_TAG in vgdl_obs:
            disc_entities.add(
                self.game.message,
                Position(*vgdl_obs[DECOY_MESSAGE_TAG]["position"]),
                self.role_2_shuffle_manual_id[DECOY_MESSAGE_TAG],
                DECOY_MESSAGE_TAG,
            )
        if DECOY_GOAL_TAG in vgdl_obs:
            disc_entities.add(
                self.game.goal,
                Position(*vgdl_obs[DECOY_GOAL_TAG]["position"]),
                self.role_2_shuffle_manual_id[DECOY_GOAL_TAG],
                DECOY_GOAL_TAG,
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
