"""
Classes that follows a gym-like interface and implements stage one of the Messenger
environment.
"""

import random
from collections import namedtuple
from pathlib import Path

import numpy as np

import messenger.envs.config as config
from messenger.envs.base import MessengerEnv, Position
from messenger.envs.DiscreteEntity import DiscreteEntities
from messenger.envs.utils import games_from_json, json_from_path

# Used to track sprites in StageOne, where we do not use VGDL to handle sprites.
Sprite = namedtuple("Sprite", ["name", "id", "position"])


class StageOne(MessengerEnv):
    def __init__(
        self,
        split,
        message_prob=0.2,
        shuffle_obs=True,
        small=False,
        # same_game_config_when_reset=False,
    ):
        """
        Stage one where objects are all immovable. Since the episode length is short and entities
        do not move, we do not use VGDL engine for efficiency.
        message_prob:
            the probability that the avatar starts with the message
        shuffle_obs:
            shuffle the observation including the text manual
        """
        super().__init__()
        self.has_messsage_prob = message_prob
        self.shuffle_obs = shuffle_obs if not small else False
        # self.shuffle_obs = shuffle_obs
        # self.overfitting_train = small
        this_folder = Path(__file__).parent

        # Get the games and manual
        games_json_path = (
            this_folder.joinpath("games.json")
            if not small
            else this_folder.joinpath("games_small.json")
        )

        text_train_name = "text_train_small.json" if small else "text_train.json"
        if "train" in split and "mc" in split:  # multi-combination games
            game_split = "train_multi_comb"
            text_json_path = this_folder.joinpath("texts", text_train_name)

        elif "train" in split and "sc" in split:  # single-combination games
            game_split = "train_single_comb"
            text_json_path = this_folder.joinpath("texts", text_train_name)

        elif "val" in split:
            game_split = "val"
            text_json_path = this_folder.joinpath("texts", "text_val.json")

        elif "test" in split:
            game_split = "test"
            text_json_path = this_folder.joinpath("texts", "text_test.json")
        else:
            raise Exception(f"Split: {split} not understood.")

        # list of Game namedtuples
        self.all_games = games_from_json(json_path=games_json_path, split=game_split)

        # we only need the immovable and unknown descriptions, so just extract those.
        descrip = json_from_path(text_json_path)

        self.descriptors = {}
        for entity in descrip:
            self.descriptors[entity] = {}
            for role in ("enemy", "message", "goal"):
                self.descriptors[entity][role] = []
                for sent in descrip[entity][role]["immovable"]:
                    self.descriptors[entity][role].append(sent)
                for sent in descrip[entity][role]["unknown"]:
                    self.descriptors[entity][role].append(sent)

        self.positions = [  # all possible entity locations
            Position(y=3, x=5),
            Position(y=5, x=3),
            Position(y=5, x=7),
            Position(y=7, x=5),
        ]
        self.avatar_start_pos = Position(y=5, x=5)
        self.avatar = None
        self.enemy = None
        self.message = None
        self.neutral = None
        self.goal = None
        self.only_one_game_config = None
        # if same_game_config_when_reset:
        # self.only_one_game_config = self.create_game_config()
        self.has_message = None
        self.game = None
        self.shuffled_pos = None
        self.shuffled_manual = None
        self.entity2manual = None

    def create_game_config(self) -> dict:
        random.seed()
        game_id = random.choice(range(len(self.all_games)))
        game = self.all_games[game_id]
        shuffled_pos = random.sample(self.positions, 4)
        has_message = random.random() < self.has_messsage_prob
        shuffled_manual, entity2manual = self._get_manual(
            game.enemy, game.message, game.goal
        )

        random_order = [0, 1, 2]
        random.shuffle(random_order)
        # this is for discrete observation
        entity2order = {
            game.enemy.id: random_order[0],
            game.message.id: random_order[1],
            game.goal.id: random_order[2],
        }

        return {
            # "game_id": game_id,
            "game": game,
            "shuffled_pos": shuffled_pos,
            "has_message": has_message,
            "shuffled_manual": shuffled_manual,
            "entity2manual": entity2manual,
            "entity2order": entity2order,
        }

    def _get_manual(self, enemy, message, goal):
        assert enemy is not None, enemy
        assert message is not None, message
        assert goal is not None, goal

        enemy_str = random.choice(self.descriptors[enemy.name]["enemy"])
        key_str = random.choice(self.descriptors[message.name]["message"])
        goal_str = random.choice(self.descriptors[goal.name]["goal"])
        manual = [enemy_str, key_str, goal_str]
        manual_ids = [0, 1, 2]
        ENTITY_IDS_LIST = [enemy.id, message.id, goal.id]

        # if self.shuffle_obs:
        # random.shuffle(manual)
        if self.shuffle_obs:
            random.shuffle(manual_ids)  # e.g [2, 0, 1] [goal, enemy, message]

        shuffled_manual = [manual[i] for i in manual_ids]
        entity_id_2_manual_id = {}
        for i in range(len(manual_ids)):
            refered_entity_id = ENTITY_IDS_LIST[manual_ids[i]]
            entity_id_2_manual_id[refered_entity_id] = i

        return shuffled_manual, entity_id_2_manual_id

    def _get_obs(self):
        entities = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        avatar = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, 1))
        for sprite in (self.enemy, self.message, self.goal):
            assert sprite is not None, sprite
            entities[sprite.position.y, sprite.position.x, 0] = sprite.id

        assert self.avatar is not None, self.avatar
        avatar[self.avatar.position.y, self.avatar.position.x, 0] = self.avatar.id
        return {"entities": entities, "avatar": avatar}

    def reset_game_config(self, **game_config):
        self.only_one_game_config = game_config

    def reset(self):
        if self.only_one_game_config is None:
            game_config = self.create_game_config()
        else:
            game_config = self.only_one_game_config

        for key, value in game_config.items():
            # assert hasattr(self, key), key
            setattr(self, key, value)

        assert self.has_message is not None, self.has_message
        if self.has_message:
            self.avatar = Sprite(
                name=config.WITH_MESSAGE.name,
                id=config.WITH_MESSAGE.id,
                position=self.avatar_start_pos,
            )
        else:  # decide whether avatar has message or not
            self.avatar = Sprite(
                name=config.NO_MESSAGE.name,
                id=config.NO_MESSAGE.id,
                position=self.avatar_start_pos,
            )

        assert self.game is not None, self.game
        enemy, message, goal = self.game.enemy, self.game.message, self.game.goal

        assert self.shuffled_pos is not None, self.shuffle_obs
        self.enemy = Sprite(name=enemy.name, id=enemy.id, position=self.shuffled_pos[0])
        self.message = Sprite(
            name=message.name, id=message.id, position=self.shuffled_pos[1]
        )
        self.goal = Sprite(name=goal.name, id=goal.id, position=self.shuffled_pos[2])
        # self.shuffled_manual, self.entity2manual = self._get_manual()

        obs = self._convert_obs_to_discrete()
        assert self.shuffled_manual is not None, self.shuffled_manual
        return obs, self.shuffled_manual

    def _convert_obs_to_discrete(self):
        if not hasattr(self, "entity2order"):
            raise AttributeError("entity2order not found")

        disc_entities = DiscreteEntities(self.entity2order)
        disc_avatar = DiscreteEntities(is_avatar=True)
        assert self.enemy is not None, self.enemy
        assert self.enemy.id in self.entity2order, self.entity2order
        assert self.entity2manual is not None, self.entity2manual

        disc_entities.add(
            self.enemy,
            self.enemy.position,
            self.entity2manual[self.enemy.id],
            self.enemy.id,
        )

        assert self.message is not None, self.message
        assert self.message.id in self.entity2order, self.entity2order
        disc_entities.add(
            self.message,
            self.message.position,
            self.entity2manual[self.message.id],
            self.message.id,
        )

        assert self.goal is not None, self.goal
        assert self.goal.id in self.entity2order, self.entity2order
        disc_entities.add(
            self.goal,
            self.goal.position,
            self.entity2manual[self.goal.id],
            self.goal.id,
        )

        assert self.avatar is not None, self.avatar
        disc_avatar.add(
            self.avatar,
            self.avatar.position,
        )
        return {
            "entity_ids": disc_entities.ids,
            "entity_pos": disc_entities.pos,
            "manual_ids": disc_entities.manual_ids,
            "avatar_ids": disc_avatar.ids,
            "avatar_pos": disc_avatar.pos,
        }

    def _move_avatar(self, action):
        assert self.avatar is not None, self.avatar

        if action == config.ACTIONS.stay:
            return

        elif action == config.ACTIONS.up:
            # print("UP")
            if self.avatar.position.y <= 0:
                return
            else:
                new_position = Position(
                    y=self.avatar.position.y - 1, x=self.avatar.position.x
                )

        elif action == config.ACTIONS.down:
            # print("DOWN")
            if self.avatar.position.y >= config.STATE_HEIGHT - 1:
                return
            else:
                new_position = Position(
                    y=self.avatar.position.y + 1, x=self.avatar.position.x
                )

        elif action == config.ACTIONS.left:
            # print('LEFT')
            if self.avatar.position.x <= 0:
                return
            else:
                new_position = Position(
                    y=self.avatar.position.y, x=self.avatar.position.x - 1
                )

        elif action == config.ACTIONS.right:
            # print('RIGHT"')
            if self.avatar.position.x >= config.STATE_WIDTH - 1:
                return
            else:
                new_position = Position(
                    y=self.avatar.position.y, x=self.avatar.position.x + 1
                )

        else:
            raise Exception(f"{action} is not a valid action.")

        self.avatar = Sprite(
            name=self.avatar.name, id=self.avatar.id, position=new_position
        )

    def _overlap(self, sprite_1, sprite_2):
        if (
            sprite_1.position.x == sprite_2.position.x
            and sprite_1.position.y == sprite_2.position.y
        ):
            return True
        else:
            return False

    def step(self, action):
        assert self.avatar is not None, self.avatar
        self._move_avatar(action)
        # obs = self._get_obs()
        obs = self._convert_obs_to_discrete()

        # when avatar touches other entities, the other doesn't disappear -> overlap check
        if self._overlap(self.avatar, self.enemy):
            return obs, -1.0, True, None  # state, reward, done, info

        if self._overlap(self.avatar, self.message):
            if self.avatar.name == config.WITH_MESSAGE.name:
                return obs, -1.0, True, None
            elif self.avatar.name == config.NO_MESSAGE.name:
                return obs, 1.0, True, None
            else:
                raise Exception("Unknown avatar name {avatar.name}")

        if self._overlap(self.avatar, self.goal):
            if self.avatar.name == config.WITH_MESSAGE.name:
                return obs, 1.0, True, None
            elif self.avatar.name == config.NO_MESSAGE.name:
                return obs, -1.0, True, None
            else:
                raise Exception("Unknown avatar name {avatar.name}")

        return obs, 0.0, False, None
