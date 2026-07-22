"""
Classes that follows a gym-like interface and implements stage two of the Messenger
environment. Uses custom assignments of entity-dynamic-role.
"""

from html import entities
import random
from pathlib import Path
from os import environ

from termcolor import cprint

# hack to stop PyGame from printing to stdout
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from messenger.envs.DiscreteEntity import DiscreteEntities
from messenger.envs.stage_two import get_entity_order
from vgdl.interfaces.gym import VGDLEnv
import numpy as np

from messenger.envs.base import MessengerEnv, Grid, Position
import messenger.envs.config as config
from messenger.envs.utils import get_game, json_from_path

GAME_FILE_TEMPLATE = """
BasicGame block_size=2
	SpriteSet
		background > Immovable randomtiling=0.9 img=oryx/floor3 hidden=True
		root >
			enemy > %s stype=avatar speed=0.5 img=oryx/alien1
			message > %s stype=avatar speed=0.5 img=oryx/bear1
			goal > %s stype=avatar speed=0.5 img=oryx/cyclop1
			wall > Immovable img=oryx/wall11
			avatar > MovingAvatar
				no_message > img=oryx/swordman1_0
				with_message > img=oryx/swordmankey1_0
	InteractionSet
		root wall > stepBack
		root EOS > stepBack
		avatar enemy > killSprite scoreChange=-1
		no_message goal > killSprite scoreChange=-1
		no_message message > transformTo stype=with_message scoreChange=0.5
		message avatar > killSprite
		goal with_message > killSprite scoreChange=1
	TerminationSet
		SpriteCounter stype=avatar limit=0 win=False
		SpriteCounter stype=goal limit=0 win=True
	LevelMapping
		. > background
		E > background enemy
		M > background message
		G > background goal
		X > background no_message
		Y > background with_message
		W > background wall
"""

LWM_SPLIT_MAP = {
    "easy": "ne_sr_and_sm",
    "medium": "se_nr_or_nm",
    "hard": "ne_nr_or_nm",
}
# dict_keys(['train', 'dev_se_nr_or_nm', 'dev_ne_sr_and_sm', 'dev_ne_nr_or_nm', 'test_se_nr_or_nm', 'test_ne_sr_and_sm', 'test_ne_nr_or_nm'])

ENEMY_TAG = "enemy.1"
MESSAGE_TAG = "message.1"
GOAL_TAG = "goal.1"
AVATAR_WITH_MESSAGE_TAG = "with_message.1"
AVATAR_NO_MESSAGE_TAG = "no_message.1"
MOVEMENT_KEYS = ["immovable", "fleeing", "chaser"]
ROLE_KEYS = [ENEMY_TAG, MESSAGE_TAG, GOAL_TAG]


class StageTwoCustom(MessengerEnv):
    """
    Full messenger environment with mobile sprites. Uses Py-VGDL as game engine.
    To avoid the need to instantiate a large number of games, (since there are
    P(12,3) = 1320 possible entity to role assignments) We apply a wrapper on top
    of the text and game state which masks the role archetypes (enemy, message goal)
    into entities (e.g. alien, knight, mage).
    """

    def __init__(
        self,
        mode: str,
        shuffle_obs: bool = False,
        fix_order: bool = False,
        split: str = "hard",
        deter_game=False,
        overfit_game=False,
        disappear=True,
        discrete_obs=True,
        is_shuffle_obs=True,
        fix_reward_2=False,
        use_role_class=True,
        use_movement_class=True,
        # same_game_config_when_reset=False,
    ):
        assert mode in ["train", "eval", "test"], mode
        assert split in ["easy", "medium", "hard"], split
        self.use_role_class = use_role_class
        self.use_movement_class = use_movement_class

        self.deter_game = deter_game
        self.overfit_game = overfit_game
        self.disappear = disappear
        self.fix_reward_2 = fix_reward_2
        if not disappear:
            cprint("Disappear is False", "red")
        self.discrete_obs = discrete_obs
        self.is_shuffle_obs = is_shuffle_obs
        if mode != "train":
            assert split is not None, "split must be provided for non-train mode"
            if mode == "eval":
                split = "dev_" + LWM_SPLIT_MAP[split]
            else:
                split = "test_" + LWM_SPLIT_MAP[split]
        else:
            split = "train"

        super().__init__()
        self.mode = mode
        self.split = split
        self.shuffle_obs = shuffle_obs  # shuffle the entity layers
        self.this_folder = Path(__file__).parent
        if self.overfit_game:
            cprint("Overfit game", "red")
            cprint("Reading from small game file", "red")
            game_file_name = "data_splits_final_with_messenger_names_small.json"
        else:
            game_file_name = "data_splits_final_with_messenger_names.json"

        games_path = self.this_folder.joinpath(
            "texts", "custom_text_splits", game_file_name
        )
        self.games = json_from_path(games_path)[self.split]

        # self.entities = random.choice(games)
        text_path = self.this_folder.joinpath(
            "texts",
            "custom_text_splits",
            "custom_text_splits_with_messenger_names.json",
        )
        self.custom_text_splits = json_from_path(text_path)

        self.init_states = [
            str(path)
            for path in self.this_folder.joinpath(
                "vgdl_files", "stage_2", "init_states"
            ).glob("*.txt")
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
        self.manual = None  # the text manual
        self.fix_order = fix_order
        self.next_init_state_idx = 0
        self.only_one_game_config = None
        self.entities = None
        self.init_state = None

    def reset_game_config(self, **game_config):
        GAME_CONFIG_KEYS = ["entities", "init_state", "manual"]
        for key in GAME_CONFIG_KEYS:
            assert key in game_config, f"{key} not in game"
        self.only_one_game_config = game_config

    def has_manual_given_game(self, game):
        for entity_name, entity_move, entity_role in game:
            manuals = self.custom_text_splits[entity_name][entity_move][entity_role][
                self.split
            ]
            if len(manuals) == 0:
                return False
        return True

    def create_game_config(self, **kwargs):
        if self.only_one_game_config is not None:
            cprint(
                "Generate a new game config. Overriding previous game config.", "yellow"
            )

        if "entities" in kwargs:
            entities = kwargs["entities"]
        else:
            game_id = random.choice(range(len(self.games)))
            entities = self.games[game_id]

        if "init_state" in kwargs:
            init_state = kwargs["init_state"]
        else:
            init_state_id = random.randrange(len(self.init_states))

        manual = [
            random.choice(
                self.custom_text_splits[entity_name][entity_move][entity_role][
                    self.split
                ]
            )
            for entity_name, entity_move, entity_role in entities
        ]

        init_state = self.init_states[init_state_id]
        return {
            "entities": entities,
            "init_state": init_state,
            "manual": manual,
        }

    def _modify_game_template_for_disappear(self, game_template: str) -> str:
        """
        Modify the game template based on the disappear setting.
        If disappear is False, replace killSprite interactions with stepBack
        to prevent entities from disappearing on non-critical interactions.
        """
        if not self.disappear:
            # remove killSprite effect in the game for non-critical interactions
            # except message and avatar, but keep killSprite for goal with_message to end game
            game_template = game_template.replace(
                "avatar enemy > killSprite scoreChange=-1",
                "avatar enemy > stepBack scoreChange=-1",
            )
            game_template = game_template.replace(
                "no_message goal > killSprite scoreChange=-1",
                "no_message goal > stepBack scoreChange=-1",
            )
            game_template = game_template.replace(
                "goal with_message > killSprite scoreChange=1",
                "goal with_message > stepBack scoreChange=1",
            )
        return game_template

    def reset(self, **kwargs):
        """
        Resets the current environment. NOTE: We remake the environment each time.
        This is a workaround to a bug in py-vgdl, where env.reset() does not
        properly reset the environment. kwargs go to get_document().

        split is one of the split keywords in custom splits
        entities is one of the games in a custom split
        """
        if self.only_one_game_config is None:
            game_config = self.create_game_config(**kwargs)
        else:
            game_config = self.only_one_game_config

        for key, value in game_config.items():
            setattr(self, key, value)

        assert self.manual is not None, "manual is None"
        entities_by_role = {}
        for entity in self.entities:
            # entity name, entity movement
            entities_by_role[entity[2]] = [entity[0], entity[1]]

        assert len(entities_by_role) == 3, f"entities_by_role={entities_by_role}"
        # get game based on entity names
        self.game = get_game(
            (
                entities_by_role["enemy"][0],
                entities_by_role["message"][0],
                entities_by_role["goal"][0],
            )
        )  # for dynamics

        # add entity movement to the game template
        this_game_template = GAME_FILE_TEMPLATE % (
            entities_by_role["enemy"][1].title(),
            entities_by_role["message"][1].title(),
            entities_by_role["goal"][1].title(),
        )
        this_game_template = self._modify_game_template_for_disappear(
            this_game_template
        )

        # args that will go into VGDL Env.
        self._envargs = {
            "game_desc": this_game_template,
            "level_file": self.init_state,
            "notable_sprites": self.notable_sprites.copy(),
            "obs_type": "objects",  # track the objects
            "block_size": 34,  # rendering block size
            "deter_game": self.deter_game,
        }
        self.env = VGDLEnv(**self._envargs)
        vgdl_obs = self.env.reset()

        # true_parsed_manual = [
        #     (entity[0], entity[1], entity[2]) for entity in self.entities
        # ]

        if self.fix_order:
            self.next_init_state_idx += 1
            if self.next_init_state_idx >= len(self.init_states):
                self.next_init_state_idx = 0

        self.role2order = get_entity_order(vgdl_obs)
        self.order2role = {v: k for k, v in self.role2order.items()}
        self.role_order_list = [self.order2role[i] for i in range(len(self.role2order))]

        # self.manual follows the order of self.entities
        shuffled_manual_order = [0, 1, 2]
        random.shuffle(shuffled_manual_order)
        self.shuffled_manual = [self.manual[i] for i in shuffled_manual_order]

        # MANUAL_ID_2_ROLE = {0: ENEMY_TAG, 1: MESSAGE_TAG, 2: GOAL_TAG}
        ORIGIN_MANUAL_ID_2_ROLE = {}
        for id, entity_info in enumerate(self.entities):
            entity_tag = entity_info[2] + ".1"
            # assert entity_tag in ENTITY_KEYS, entity_tag
            ORIGIN_MANUAL_ID_2_ROLE[id] = entity_tag

        self.role2shuffled_manual_id = {
            role: shuffled_manual_order.index(origin_manual_id)
            for origin_manual_id, role in ORIGIN_MANUAL_ID_2_ROLE.items()
        }
        self.shuffled_manual_ids = shuffled_manual_order
        assert self.manual is not None, "self.manual is None"
        shuffled_manual = [self.manual[i] for i in shuffled_manual_order]

        self.manual_ids = [
            self.role2shuffled_manual_id[k] for k in self.role_order_list
        ]
        role2movement = {}
        for entity_info in self.entities:
            entity_role = entity_info[2] + ".1"
            # assert entity_role in ENTITY_KEYS, entity_role
            role2movement[entity_role] = entity_info[1]

        self.manual_ids = np.array(self.manual_ids)
        self.movement_classes = [role2movement[k] for k in self.role_order_list]
        self.movement_ids = [MOVEMENT_KEYS.index(k) for k in self.movement_classes]

        self.role_classes = self.role_order_list
        self.role_ids = [ROLE_KEYS.index(k) for k in self.role_classes]

        if self.discrete_obs:
            obs = self._convert_obs_to_discrete(vgdl_obs)
        else:
            obs = self._convert_obs(vgdl_obs)

        self.true_parsed_manual = self.entities

        if self.use_role_class:
            obs["role"] = np.array(self.role_ids)
        if self.use_movement_class:
            obs["movement"] = np.array(self.movement_ids)

        return obs, shuffled_manual

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

        if AVATAR_NO_MESSAGE_TAG in vgdl_obs:
            """
            Due to a quirk in VGDL, the avatar is no_message if it starts as no_message
            even if the avatar may have acquired the message at a later point.
            To check, if it has a message, check that the class vector corresponding to
            with_message is == 1.
            """
            avatar_pos = Position(*vgdl_obs[AVATAR_NO_MESSAGE_TAG]["position"])
            # with_key is last position, see self.notable_sprites
            if vgdl_obs[AVATAR_NO_MESSAGE_TAG]["class"][-1] == 1:
                avatar = config.WITH_MESSAGE
            else:
                avatar = config.NO_MESSAGE

        elif AVATAR_WITH_MESSAGE_TAG in vgdl_obs:
            # this case only occurs if avatar begins as with_message at start of episode
            avatar_pos = Position(*vgdl_obs[AVATAR_WITH_MESSAGE_TAG]["position"])
            avatar = config.WITH_MESSAGE

        else:  # the avatar is not in observation, so is probably dead
            return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

        avatar_locs.add(avatar, avatar_pos)  # if not dead, add it.

        return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

    def step(self, action):
        # assert action is not nan using np.isnan
        assert not np.isnan(action).any()
        vgdl_obs, reward, done, info = self.env.step(action)

        if self.discrete_obs:
            obs = self._convert_obs_to_discrete(vgdl_obs)
        else:
            obs = self._convert_obs(vgdl_obs)

        if self.use_role_class:
            obs["role"] = np.array(self.role_ids)
        if self.use_movement_class:
            obs["movement"] = np.array(self.movement_ids)

        if not self.disappear:
            done = done or (reward != 0 and reward != 0.5)

            if reward == -2:
                reward = -1
            elif reward == 2:
                reward = 1
            elif reward == -0.5:
                reward = -1

            if done and reward != 1:
                reward = -1

        return obs, reward, done, info

    def _convert_obs_to_discrete(self, vgdl_obs):
        disc_entities = DiscreteEntities(self.role2order)
        disc_avatar = DiscreteEntities(is_avatar=True)
        # if self.use_role_class:
        #     role_entiites = DiscreteEntities(self.role2order)

        if ENEMY_TAG in vgdl_obs:
            disc_entities.add(
                self.game.enemy,
                Position(*vgdl_obs[ENEMY_TAG]["position"]),
                self.role2shuffled_manual_id[ENEMY_TAG],
                ENEMY_TAG,
            )
            # if self.use_role_class:
            #     role_entiites.add(
            #         ROLE_KEYS.index(ENEMY_TAG),
            #         Position(*vgdl_obs[ENEMY_TAG]["position"]),
            #         self.role2shuffled_manual_id[ENEMY_TAG],
            #     )

        if MESSAGE_TAG in vgdl_obs:
            disc_entities.add(
                self.game.message,
                Position(*vgdl_obs[MESSAGE_TAG]["position"]),
                self.role2shuffled_manual_id[MESSAGE_TAG],
                MESSAGE_TAG,
            )
            # if self.use_role_class:
            #     role_entiites.add(
            #         ROLE_KEYS.index(MESSAGE_TAG),
            #         Position(*vgdl_obs[MESSAGE_TAG]["position"]),
            #         self.role2shuffled_manual_id[MESSAGE_TAG],
            #     )

        if GOAL_TAG in vgdl_obs:
            disc_entities.add(
                self.game.goal,
                Position(*vgdl_obs[GOAL_TAG]["position"]),
                self.role2shuffled_manual_id[GOAL_TAG],
                GOAL_TAG,
            )
            # if self.use_role_class:
            #     role_entiites.add(
            #         ROLE_KEYS.index(GOAL_TAG),
            #         Position(*vgdl_obs[GOAL_TAG]["position"]),
            #         self.role2shuffled_manual_id[GOAL_TAG],
            #     )

        if AVATAR_NO_MESSAGE_TAG in vgdl_obs:
            """
            Due to a quirk in VGDL, the avatar is no_message if it starts as no_message
            even if the avatar may have acquired the message at a later point.
            To check, if it has a message, check that the class vector corresponding to
            with_message is == 1.
            """
            avatar_pos = Position(*vgdl_obs[AVATAR_NO_MESSAGE_TAG]["position"])
            # with_key is last position, see self.notable_sprites
            if vgdl_obs[AVATAR_NO_MESSAGE_TAG]["class"][-1] == 1:
                avatar = config.WITH_MESSAGE
            else:
                avatar = config.NO_MESSAGE

        elif AVATAR_WITH_MESSAGE_TAG in vgdl_obs:
            # this case only occurs if avatar begins as with_message at start of episode
            avatar_pos = Position(*vgdl_obs[AVATAR_WITH_MESSAGE_TAG]["position"])
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
        res = {
            "entity_ids": disc_entities.ids,
            "entity_pos": disc_entities.pos,
            "manual_ids": disc_entities.manual_ids,
            "avatar_ids": disc_avatar.ids,
            "avatar_pos": disc_avatar.pos,
        }

        # if self.use_role_class:
        #     res["role"] = role_entiites.ids

        return res
