# %%
from messenger.envs.config import ENTITY_IDS
import numpy as np
import random
import pathlib
import random
from termcolor import cprint
from ledwm.embodied.core.base import Env
import numpy as np
from gym import spaces
import sys
from ledwm import constants
from ledwm.embodied.envs.sentence_embedding_cache import (
    load_sentence_embeddings,
    should_log_sentence_embeddings,
)
from ledwm.logging_setup import logger as event_logger

sys.path.append("messenger-emma")


# MOVEMENT2ID = {"immovable": 0, "fleeing": 1, "chaser": 2}
# ROLE2ID = {"enemy.1": 0, "message.1": 1, "goal.1": 2}

NUM_SENTS = {
    "easy": 3,
    "medium": 3,
    "hard": 3,
}
NUM_ENTITIES = {
    "easy": 3,
    "medium": 3,
    "hard": 3,
}
DEAD_ID = 0
DEAD_POS = 10


MAX_SHAPE_GRID = 10
MAX_GAMES = {
    "easy": 1536,
    "medium": 1536,
    "hard": 1536,
}
HIST_LEN = 33
MIN_HIST_LEN = 2
EPSILON = 1e-6


def _data_dir():
    return pathlib.Path(__file__).parent / "data"


class LWMSent(Env):
    def __init__(
        self,
        task,
        length,
        mode="train",
        size=(16, 16),
        load_embeddings=True,
        vis=False,
        use_sent_ids=True,
        t5_sent=False,
        use_time_step=True,
        small_image=False,
        use_lang=True,  # use_reading
        enforce_num_pixels=False,
        remove_image=True,
        aug=True,
        sent_dim=constants.SENT_DIM,
        entity_track=False,
        hist_len=HIST_LEN,
        model_sent="mpnet",
        debug_pos_gt=False,
        use_movement_class=True,
        mask_future_steps_dp=True,
        deter_game=False,
        overfit_game=False,
        disappear=True,
        fix_reward_2=False,
        use_role=False,
    ):
        assert task in ("easy", "medium", "hard"), task
        assert mode in ("train", "eval", "test"), mode
        self.use_role = use_role
        self.deter_game = deter_game
        self.overfit_game = overfit_game
        self.mask_future_steps_dp = mask_future_steps_dp
        self.debug_pos_gt = debug_pos_gt
        self.use_movement_class = use_movement_class
        self.model_sent = model_sent
        self.entity_track = entity_track
        self.hist_len = hist_len

        self.aug = aug if mode == "train" and task == "s1" else False
        self.remove_image = remove_image
        self.small_image = small_image
        self.use_time_step = use_time_step
        self.use_sent_ids = use_sent_ids
        self.use_lang = use_lang
        self.enforce_num_pixels = enforce_num_pixels
        self.t5_sent = t5_sent
        self.sent_dim = sent_dim
        self.fix_reward_2 = fix_reward_2
        if self.t5_sent:
            pass

        assert length > 1, length
        event_logger.debug(
            f"env.config | type=messenger_lwm | task={task} | mode={mode} | "
            f"length={length} | visibility={str(vis).lower()} | "
            f"use_sent_ids={str(use_sent_ids).lower()}"
        )

        from messenger.envs.stage_two_custom import StageTwoCustom

        self.task = task
        self._env = StageTwoCustom(
            mode,
            split=task,
            deter_game=deter_game,
            overfit_game=overfit_game,
            disappear=disappear,
            fix_reward_2=fix_reward_2,
            # use_role_class=use_role,
            # use_movement_class=use_movement_class,
            # same_game_config_when_reset=same_game_config_when_reset,
        )

        self.num_sents = NUM_SENTS[task]
        from ledwm.embodied.envs import from_gym

        self.wrappers = [from_gym.FromGym]
        if not self.remove_image:
            from ledwm.embodied.core import wrappers

            self.wrappers.append(
                # Pad image to multiple of 2
                lambda e: wrappers.PadImage(
                    e, "small_image" if self.small_image else "image", size
                )  # type: ignore
            )

        self.max_token_seqlen = 36  # per manual sentence
        self.manual = None
        self.tokens = []
        self.current_sentence = None
        self._step = 0
        self.length = length
        self.num_all_entities = constants.NUM_ALL_ENTITIES
        self.num_entities_task = NUM_ENTITIES[task]

        self.grid_size = size
        self.vis = vis
        self.mode = mode

        if load_embeddings and self.fname().exists():
            self._load_mean_sent_embeds()
        else:
            raise ValueError("No embeddings found", self.fname())

    # s1_train_sent_embeddings
    def fname(self):
        if self.t5_sent:
            return _data_dir() / f"{self.task}_{self.mode}_sent_embeddings.pkl"
        else:
            mode = "train_eval_test"
            return _data_dir() / "messenger" / f"{mode}_lwm_{self.model_sent}.pkl"

    def _load_mean_sent_embeds(self):
        fname = self.fname()
        self.sent2id, self.id2sentemb, cache = load_sentence_embeddings(
            fname, self.t5_sent
        )
        if should_log_sentence_embeddings(cache):
            event_logger.debug(
                f"env.sentence_embeddings_loaded | count={len(self.id2sentemb)} | "
                f"shape={self.id2sentemb.shape} | path={fname} | "
                f"storage={'mmap' if cache.memory_mapped else 'private'}"
            )

    def _update_pos_entity_avatar(self, obs):
        if self.use_role:
            entity_pos = np.concatenate(
                [obs["entity_pos"], obs["role"][:, None]], axis=-1
            )
        else:
            entity_pos = np.concatenate(
                [obs["entity_pos"], obs["entity_ids"][:, None]], axis=-1
            )

        avatar_pos = np.concatenate(
            [obs["avatar_pos"], obs["avatar_ids"][:, None]], axis=-1
        )
        return {
            "entity_pos": entity_pos.astype(int),
            "avatar_pos": avatar_pos.astype(int),
        }

    def _get_entity_tracking(self, obs, reset=False):  # hist_len, N, 3
        entity_pos = np.copy(obs["entity_pos"])[:, :2]
        avatar_pos = np.copy(obs["avatar_pos"])[:, :2]
        if reset:
            assert self._step == 0, self._step
            # fill with -1
            self.entity_pos_hist = np.ones(
                (self.hist_len, self.num_entities_task, 2)
            ) * (-1)
            self.avatar_pos_hist = np.ones((self.hist_len, 1, 2)) * (-1)
            self.entity_vel_hist = np.zeros((self.hist_len, self.num_entities_task, 2))

            self.entity_pos_hist[0] = entity_pos
            self.avatar_pos_hist[0] = avatar_pos

        else:
            assert self._step > 0, self._step

            previous_entity_pos = self.entity_pos_hist[self._step - 1]
            if DEAD_POS in entity_pos:
                for i in range(self.num_entities_task):
                    if DEAD_POS in entity_pos[i]:
                        entity_pos[i] = previous_entity_pos[i]

            self.entity_vel_hist[self._step] = entity_pos - previous_entity_pos
            self.entity_pos_hist[self._step] = entity_pos

            if DEAD_POS in avatar_pos:
                previous_avatar_pos = self.avatar_pos_hist[self._step - 1]
                avatar_pos = previous_avatar_pos
            self.avatar_pos_hist[self._step] = avatar_pos

        res = {}
        # res["entity_pos_hist"] = np.array(self.entity_pos_hist)
        # res["avatar_pos_hist"] = np.array(self.avatar_pos_hist)

        # res = {
        #     "pos_vec_hist": np.array(self.entity_pos_hist)
        #     - np.array(self.avatar_pos_hist),
        #     "entity_vec_hist": np.array(self.entity_vel_hist),
        # }  # (hist_len, N, 2)

        # res["pos_vec_hist"] = normed_pos_vec_hist.astype(np.float32)
        # res["entity_vec_hist"] = entity_vec_hist.astype(np.float32)

        # dot product between pos_vec_hist and entity_vec_hist

        # turn (hist_len, N, 2) -> (N, hist_len, 2) -> (N, hist_len * 2)
        res = {k: v.transpose(1, 0, 2).reshape(v.shape[1], -1) for k, v in res.items()}

        pos_vec_hist = np.array(self.entity_pos_hist) - np.array(self.avatar_pos_hist)
        entity_vec_hist = np.array(self.entity_vel_hist)
        norm = np.linalg.norm(pos_vec_hist, axis=-1, keepdims=True)
        norm = np.where(norm < EPSILON, 1, norm)
        normed_pos_vec_hist = np.where(
            norm < EPSILON, pos_vec_hist, pos_vec_hist / norm
        )

        # assert norm doesn't have nan
        assert not np.isnan(normed_pos_vec_hist).any(), f"{normed_pos_vec_hist=}"
        pos_vec_hist = normed_pos_vec_hist.astype(np.float32)  # (hist_len, N, 2)
        res["dp"] = np.sum(pos_vec_hist * entity_vec_hist, axis=-1).T  # N, hist_len
        if self.mask_future_steps_dp:
            res["dp"][:, self._step + 1 :] = -1
        return res

    # def _get_movement_class(self, obs):
    #     assert self.movement_classes is not None, self.movement_classes
    #     res = np.zeros((self.num_entities_task))
    #     for i in range(self.num_entities_task):
    #         res[i] = MOVEMENT2ID[self.movement_classes[i]]
    #     return res.astype(np.int32)

    # def _get_role_class(self, obs):
    #     assert self.role_classes is not None, self.role_classes
    #     res = np.zeros((self.num_entities_task))
    #     for i in range(self.num_entities_task):
    #         res[i] = ROLE2ID[self.role_classes[i]]
    #     return res.astype(np.int32)

    def _symbolic_to_multihot(self, obs):
        # (h, w, Ne + 1) (+1: agent)
        layers = np.concatenate((obs["entities"], obs["avatar"]), axis=-1).astype(int)
        new_ob = np.maximum.reduce(
            [
                np.eye(self.num_all_entities)[layers[..., i]]
                for i in range(layers.shape[-1])
            ]
        )
        new_ob[:, :, 0] = 0
        return new_ob

    @property
    def observation_space(self):
        # from messenger.envs.config import STATE_HEIGHT, STATE_WIDTH
        # original_grid_size = (STATE_HEIGHT, STATE_WIDTH)
        obs_space = {}
        obs_space["game_id"] = spaces.Box(
            low=0, high=MAX_GAMES[self.task], shape=(), dtype=int
        )

        if not self.remove_image:
            obs_space["image"] = spaces.Box(
                low=0,
                high=1,
                shape=(
                    *self.grid_size,
                    self.num_all_entities,
                ),
            )

        if self.entity_track:
            obs_space["entity_pos_hist"] = spaces.Box(
                low=-1,
                high=10,  # max(16, 16, 17)
                # shape=(HIST_LEN, self.num_entities_task, 2),
                shape=(self.num_entities_task, 2 * HIST_LEN),
            )
            obs_space["avatar_pos_hist"] = spaces.Box(
                low=-1,
                high=10,
                # shape=(HIST_LEN, 1, 2),
                shape=(1, 2 * HIST_LEN),
            )

            # obs_space["entity_vec_hist"] = spaces.Box(
            #     low=-10,
            #     high=10,
            #     shape=(self.num_entities_task, 2 * HIST_LEN),
            # )
            # obs_space["pos_vec_hist"] = spaces.Box(
            #     low=-1,
            #     high=1,
            #     shape=(self.num_entities_task, 2 * HIST_LEN),
            # )
            obs_space["dp"] = spaces.Box(
                low=-1,
                high=3,
                shape=(self.num_entities_task, HIST_LEN),
            )

            obs_space["entity_pos"] = spaces.Box(
                low=0,
                high=10,
                shape=(self.num_entities_task, 3),
            )
            # obs_space["entity_ids_hist"] = spaces.Box(
            #     low=0,
            #     high=17,  # max(16, 16, 17)
            #     shape=(
            #         3,
            #         self.num_entities_task,
            #     ),
            # )

            # obs_space["dist"] = spaces.Box(
            #     low=0, high=20, shape=(self.num_entities_task, HIST_LEN)
            # )
            # obs_space["avatar_ids_hist"] = spaces.Box(
            #     low=0,
            #     high=17,
            #     shape=(3, 1),
            # )

        obs_space.update(
            {
                "manual_ids": spaces.Box(
                    low=-1,  # when entity id = 0 is dead
                    high=self.num_sents,
                    shape=(self.num_entities_task,),
                ),
            }
        )
        obs_space.update(
            {
                "entity_ids": spaces.Box(
                    low=0,
                    high=self.num_all_entities,
                    shape=(self.num_entities_task,),
                ),  # H, W, Ne
                "entity_pos": spaces.Box(
                    low=0,
                    high=17,  # max(16, 16, 17)
                    shape=(self.num_entities_task, 3),
                ),  # H, W, Ne, 3 (10, 10, N in image)
                "avatar_ids": spaces.Box(
                    low=0,
                    high=self.num_all_entities,
                    shape=(1,),
                ),  # H, W, 1
                "avatar_pos": spaces.Box(
                    low=0,
                    high=17,
                    shape=(1, 3),
                ),  # H, W, 1, 3 (10, 10, N in image)
            }
        )
        # if self.enforce_num_pixels:
        #     obs_space["num_pixels"] = spaces.Box(
        #         low=0, high=self.num_sents + 1, shape=(), dtype=int
        #     )

        if self.use_time_step:
            obs_space["time_step"] = spaces.Box(
                low=0,
                high=self.length,
                shape=(),
                dtype=int,
            )

        if self.use_lang:
            if self.use_sent_ids:
                obs_space.update(
                    {
                        "sent_ids": spaces.Box(
                            0, 50000, shape=(self.num_sents,), dtype=int
                        ),
                    }
                )

        obs_space["movement"] = spaces.Box(
            low=0,
            high=3,
            shape=(self.num_entities_task,),
        )

        obs_space["role"] = spaces.Box(
            low=0,
            high=3,
            shape=(self.num_entities_task,),
        )

        return spaces.Dict(obs_space)

    @property
    def action_space(self):
        return self._env.action_space

    def _embed_mean_sent(self, sent, return_id=False):
        # else:
        if return_id:
            return self.sent2id[sent]
        else:
            mean_sent = self.id2sentemb[self.sent2id[sent]]
            return mean_sent

    def get_sent_embeds(self):
        sent_embed = []
        for sent in self.manual_sentences:
            sent_embed.append(self._embed_mean_sent(sent))

        # turn sent_embed: list of [(S, dim)] -> (S, dim)
        sent_embed = np.stack(sent_embed, axis=0)
        assert sent_embed.shape == (self.num_sents, self.sent_dim), sent_embed.shape
        return sent_embed

    def get_sent_ids(self):
        sent_embed_ids = []
        for sent in self.manual_sentences:
            sent_embed_ids.append(self._embed_mean_sent(sent, return_id=True))
        return np.array(sent_embed_ids).astype(np.int32)

    def get_mean_over_sents(self):
        if hasattr(self, "mean_sent"):
            return self.mean_sent
        else:
            self.mean_sent = self.get_sent_embeds().mean(0).astype(np.float32)
            return self.mean_sent

    def reset_game_config(self, **kwargs):
        return self._env.reset_game_config(**kwargs)

    def create_game_config(self, **kwargs):
        return self._env.create_game_config(**kwargs)

    def has_manual_given_game(self, game):
        return self._env.has_manual_given_game(game)

    @property
    def env_cache(self):
        if self.use_sent_ids:
            assert len(self.sent2id) == len(self.id2sentemb), (
                f"{len(self.sent2id)=} {len(self.id2sentemb)=}"
            )
            # turn self.mean_sent_cache: list of np.array -> np.array
            return {
                "sent_embed": np.array(self.id2sentemb),
                "sent_ids": self.sent2id,
            }
        else:
            return None

    def reset(self):
        self._step = 0
        obs, self.manual_sentences = self._env.reset()
        self.true_parsed_manual = self._env.true_parsed_manual

        for entity_info in self.true_parsed_manual:
            entity_id = ENTITY_IDS[entity_info[0]]
            assert entity_id in obs["entity_ids"], (
                f"Entity {entity_id} not in {obs['entity_ids']}"
            )

        self.manual = "</s>" + "</s>".join([x.strip() for x in self.manual_sentences])
        # to feed gt role to wm
        # if self.use_movement_class:
        #     self.movement_classes = self._env.movement_classes
        # if self.use_role:
        #     self.role_classes = self._env.role_classes

        if self.use_lang:
            if self.use_sent_ids:
                self.send_ids = self.get_sent_ids()
                obs: dict
                obs.update({"sent_ids": self.get_sent_ids()})
            else:
                self.sent_embed = self.get_sent_embeds()
                obs.update({"sent_embed": self.sent_embed})
            obs.update(
                {
                    "log_language_info": self.manual,
                }
            )

        # if not self.remove_image:
        #     if self.small_image:
        #         obs["small_image"] = np.concatenate(
        #             (obs["entities"], obs["avatar"]), axis=-1
        #         ).astype(int)
        #     else:
        #         obs["image"] = self._symbolic_to_multihot(obs)

        obs.update(self._update_pos_entity_avatar(obs))

        if self.entity_track:
            obs.update(self._get_entity_tracking(obs, True))

        # if self.use_movement_class:
        #     self.movement = self._get_movement_class(obs)
        #     obs["movement"] = self.movement

        # if self.use_role:
        #     self.role = self._get_role_class(obs)
        #     obs["role"] = self.role

        # if "manual_ids" not in obs:
        #     assert self.task in ["s1", "s2"]
        #     obs["manual_ids"] = self._get_manual_ids(obs)

        # if self.vis:
        #     # assert not self.remove_image
        #     if "image" in obs:
        #         image = obs["image"]
        #     else:
        #         image = self._symbolic_to_multihot(obs)
        #     obs["log_image"] = self.log_image(image, -1, 0, False)

        if self.use_time_step:
            obs["time_step"] = self._step

        if self.enforce_num_pixels:
            obs["num_pixels"] = self.num_sents + 1

        if "entities" in obs:
            del obs["entities"]
        if "avatar" in obs:
            del obs["avatar"]

        return obs  # no reward here

    def step(self, action):
        self._step += 1
        obs, reward, done, info = self._env.step(action)
        obs: dict
        info = info or {}
        # obs["game_id"] = self.game_id

        # obs["manual_entity_ids"] = self.manual_entity_ids
        if self.use_lang:
            if self.use_sent_ids:
                obs["sent_ids"] = self.send_ids
            else:
                obs["sent_embed"] = self.sent_embed
            obs.update(
                {
                    "log_language_info": self.manual,
                    # "mean_sent_embed": self.get_mean_over_sents(),
                    #      "log_tokens": self.tokens,
                }
            )

        # if not self.remove_image:
        #     if self.small_image:
        #         obs["small_image"] = np.concatenate(
        #             (obs["entities"], obs["avatar"]), axis=-1
        #         ).astype(int)
        #     else:
        #         obs["image"] = self._symbolic_to_multihot(obs)

        obs.update(self._update_pos_entity_avatar(obs))

        if self.entity_track:
            obs.update(self._get_entity_tracking(obs))

        # if self.use_movement_class:
        #     obs["movement"] = self.movement

        # if self.use_role:
        #     self.role = self._get_role_class(obs)
        #     obs["role"] = self.role

        # if "manual_ids" not in obs:
        # assert self.task in ["s1", "s2"]
        # obs["manual_ids"] = self._get_manual_ids(obs)
        # info.update(
        #     {
        #         "entities": obs["entities"],
        #         "avatar": obs["avatar"],
        #     }
        # )

        # if self.enforce_num_pixels:
        # obs["num_pixels"] = self.num_sents + 1

        if self.use_time_step:
            obs["time_step"] = self._step

        # timeout
        if self._step >= self.length:
            done = True
            reward = -1

        # if self.vis:
        #     #     assert not self.remove_image
        #     if "image" in obs:
        #         image = obs["image"]
        #     else:
        #         image = self._symbolic_to_multihot(obs)
        #     obs["log_image"] = self.log_image(image, self._step, action, reward, done)

        if "entities" in obs:
            del obs["entities"]
        if "avatar" in obs:
            del obs["avatar"]

        return obs, reward, done, info

    # def _get_manual_ids(self, obs):
    #     return np.array(
    #         [self.entity2manual[x] if x > 0 else -1 for x in obs["entity_ids"]]
    #     )


# %%
from collections import deque, namedtuple

NUM_TIME_STEPS = {
    "s1": 5,
    "s2": 65,
    "s3": 129,
    "easy": 33,
    "medium": 33,
    "hard": 33,
    "lwm": 33,
}
ACTIONS = namedtuple("Actions", "up down left right stay")(0, 1, 2, 3, 4)


# def test_aug():
#     import sys

#     sys.path.append("messenger-emma/")
#     sys.path.append("ledwm")
#     env = MessengerSent("s2", mode="train")
#     obs = env.reset()
#     print(env._env.entity2manual)
#     print(env._env.entity2order)
#     for k, v in env._env.vgdl_obs.items():
#         print(k, v)

#     for k, v in obs.items():
#         if hasattr(v, "shape"):
#             print(k, v.shape)
#         else:
#             print(k, v)

#     print(f'{obs["entity_pos"]}')
#     print(f'{obs["avatar_pos"]}')
#     print(f'{obs["entity_ids"]}')
#     for id in obs["entity_ids"]:
#         print(id, IDX_TO_ENTITY_NAME[id])
#     print("")

#     # action = 2
#     # print(f"{action=}")
#     # obs, reward, done, info = env.step(action)
#     # print()
#     # print(f'{obs["entity_pos"]=}')
#     # print(f'{obs["avatar_pos"]=}')

#     for t in range(64):
#         # random action
#         action = random.choice(range(5))
#         obs, reward, done, info = env.step(action)
#         if done:
#             break
#         print(f"{action=}")
#         print(f'{obs["entity_ids"]}')
#         print(f'{obs["entity_pos"]}')
#         print(f'{obs["avatar_pos"]}')


# test_aug()


# def flip_horizontally(entity_pos, avatar_pos):
#     entity_pos[:, 1] = MAX_SHAPE_GRID - entity_pos[:, 1]
#     avatar_pos[:, 1] = MAX_SHAPE_GRID - avatar_pos[:, 1]
#     return entity_pos, avatar_pos


# def flip_vertically(entity_pos, avatar_pos):
#     entity_pos[:, 0] = MAX_SHAPE_GRID - entity_pos[:, 0]
#     avatar_pos[:, 0] = MAX_SHAPE_GRID - avatar_pos[:, 0]
#     return entity_pos, avatar_pos


# def flip_both(entity_pos, avatar_pos):
#     entity_pos, avatar_pos = flip_horizontally(entity_pos, avatar_pos)
#     entity_pos, avatar_pos = flip_vertically(entity_pos, avatar_pos)
#     return entity_pos, avatar_pos


# def rotate_90(entity_pos, avatar_pos):
#     entity_pos[:, [0, 1]] = np.stack(
#         [entity_pos[:, 1], MAX_SHAPE_GRID - entity_pos[:, 0]], axis=-1
#     )
#     avatar_pos[:, [0, 1]] = np.stack(
#         [avatar_pos[:, 1], MAX_SHAPE_GRID - avatar_pos[:, 0]], axis=-1
#     )
#     return entity_pos, avatar_pos


# def rotate_180(entity_pos, avatar_pos):
#     entity_pos[:, [0, 1]] = MAX_SHAPE_GRID - entity_pos[:, [0, 1]]
#     avatar_pos[:, [0, 1]] = MAX_SHAPE_GRID - avatar_pos[:, [0, 1]]
#     return entity_pos, avatar_pos


# def rotate_270(entity_pos, avatar_pos):
#     entity_pos[:, [0, 1]] = np.stack(
#         [MAX_SHAPE_GRID - entity_pos[:, 1], entity_pos[:, 0]], axis=-1
#     )
#     avatar_pos[:, [0, 1]] = np.stack(
#         [MAX_SHAPE_GRID - avatar_pos[:, 1], avatar_pos[:, 0]], axis=-1
#     )
#     return entity_pos, avatar_pos


# def shift_x(entity_pos, avatar_pos, delta):

#     entity_pos[:, 0] = entity_pos[:, 0] + delta
#     avatar_pos[:, 0] = avatar_pos[:, 0] + delta
#     return entity_pos, avatar_pos


# def shift_y(entity_pos, avatar_pos, delta):
#     entity_pos[:, 1] = entity_pos[:, 1] + delta
#     avatar_pos[:, 1] = avatar_pos[:, 1] + delta
#     return entity_pos, avatar_pos


# def shift_xy(entity_pos, avatar_pos, delta):
#     entity_pos, avatar_pos = shift_x(entity_pos, avatar_pos, delta)
#     entity_pos, avatar_pos = shift_y(entity_pos, avatar_pos, delta)
#     return entity_pos, avatar_pos


# def no_augmentation(entity_pos, avatar_pos):
#     return entity_pos, avatar_pos
