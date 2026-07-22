# %%
import pathlib
import sys


def _checkout_root():
    return pathlib.Path(__file__).parents[3]


def _add_checkout_paths():
    root = _checkout_root()
    for path in (root / "messenger-emma", root / "ledwm"):
        path = str(path)
        if path not in sys.path:
            sys.path.append(path)


_add_checkout_paths()

import random
from collections import namedtuple
import numpy as np
from messenger.envs.TwoEnvWrapper import TwoEnvWrapper
from termcolor import cprint
from tqdm import tqdm
from ledwm.common import IDX_TO_ENTITY_NAME
from ledwm.embodied.core.base import Env
from gym import spaces


from ledwm import constants
from ledwm.embodied.envs.sentence_embedding_cache import (
    load_sentence_embeddings,
    should_log_sentence_embeddings,
)
from ledwm.logging_setup import logger as event_logger

MOVEMENT_2_ID = {"immovable": 0, "fleeing": 1, "chaser": 2}
ID_2_MOVEMENT = {v: k for k, v in MOVEMENT_2_ID.items()}

NUM_SENTS = {
    "s1": 3,
    "s2": 3,
    "s3": 6,
    "easy": 3,
    "medium": 3,
    "hard": 3,
}
NUM_ENTITIES = {
    "s1": 3,
    "s2": 3,
    "s3": 5,
    "easy": 3,
    "medium": 3,
    "hard": 3,
}
DEAD_ID = 0
DEAD_POS = 10


MAX_SHAPE_GRID = 10
MAX_GAMES = {
    "s1": 44,
    "s2": 44,
    "s3": 44,
    "easy": 1536,
    "medium": 1536,
    "hard": 1536,
}
HIST_LEN = 33
MIN_HIST_LEN = 2
EPSILON = 1e-6


def _data_dir():
    return pathlib.Path(__file__).parent / "data"


class MessengerSent(Env):
    def __init__(
        self,
        task,
        length,
        mode="train",
        size=(16, 16),
        vis=False,
        use_sent_ids=True,
        t5_sent=False,
        read=False,
        read_steps=0,
        use_time_step=False,
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
        use_movement_class=False,
        deter_game=False,
        mask_future_steps_dp=True,
    ):
        assert task in ("s1", "s2", "s3") or task in ("easy", "medium", "hard")
        assert mode in ("train", "eval", "test", "test-se")
        # self.same_game_config_when_reset = same_game_config_when_reset
        self.debug_pos_gt = debug_pos_gt
        self.use_movement_class = use_movement_class
        self.model_sent = model_sent
        self.entity_track = entity_track
        self.hist_len = hist_len
        self.mask_future_steps_dp = mask_future_steps_dp
        self.aug = aug if mode == "train" and task == "s1" else False
        self.remove_image = remove_image
        self.small_image = small_image
        self.use_time_step = use_time_step
        self.use_sent_ids = use_sent_ids
        self.use_lang = use_lang
        self.use_read = read
        self.enforce_num_pixels = enforce_num_pixels
        if self.use_lang:
            self.use_read = False

        self.max_read_steps = read_steps
        if self.use_read:
            assert self.max_read_steps > 0, self.max_read_steps

        self.t5_sent = t5_sent
        self.sent_dim = sent_dim
        if self.t5_sent:
            pass

        assert length > 1, length
        event_logger.debug(
            f"env.config | type=messenger | task={task} | mode={mode} | "
            f"length={length} | visibility={str(vis).lower()} | "
            f"use_sent_ids={str(use_sent_ids).lower()}"
        )

        self.task = task
        if task == "s1":
            # mmode = "train" if mode == "train" else "val"
            if mode == "train":
                self._env = TwoEnvWrapper(
                    stage=1,
                    split_1="train-mc",
                    split_2="train-sc",
                    prob_env_1=0.75,
                    # small=small,
                )
            elif mode == "eval":
                from messenger.envs.stage_one import StageOne

                self._env = StageOne(split="val")
            else:
                assert mode == "test", mode
                from messenger.envs.stage_one import StageOne

                self._env = StageOne(split="test")

        elif task == "s2":
            if mode == "train":
                # kwargs=dict(stage=2, split_1="train_mc", split_2="train_sc", prob_env_1=0.75),
                self._env = TwoEnvWrapper(
                    stage=2,
                    split_1="train-mc",
                    split_2="train-sc",
                    prob_env_1=0.75,
                    deter_game=deter_game,
                )
            elif mode == "eval":
                from messenger.envs.stage_two import StageTwo

                self._env = StageTwo(
                    split="val",
                    deter_game=deter_game,
                    # same_game_config_when_reset=same_game_config_when_reset,
                )
            else:
                assert "test" in mode
                from messenger.envs.stage_two import StageTwo

                self._env = StageTwo(
                    split=mode,
                    deter_game=deter_game,
                    # same_game_config_when_reset=same_game_config_when_reset,
                )

        elif task == "s3":
            if mode == "train":
                self._env = TwoEnvWrapper(
                    stage=3,
                    split_1="train-mc",
                    split_2="train-sc",
                    prob_env_1=0.75,
                )
            elif mode == "eval":
                from messenger.envs.stage_three import StageThree

                self._env = StageThree(
                    split="val",
                    # same_game_config_when_reset=same_game_config_when_reset,
                )
            else:
                assert mode == "test", mode
                from messenger.envs.stage_three import StageThree

                self._env = StageThree(
                    split="test",
                    # same_game_config_when_reset=same_game_config_when_reset,
                )

        elif "lwm" in task:
            from messenger.envs.stage_two_custom import StageTwoCustom

            split = task.split("-")[-1]
            self._env = StageTwoCustom(mode, split=split)
        else:
            raise ValueError(f"{task=}")

        self.num_sents = NUM_SENTS[task]
        from ledwm.embodied.envs import from_gym
        from ledwm.embodied.core import wrappers

        self.wrappers = [from_gym.FromGym]
        if not self.remove_image:
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
        self._init_obs = None
        self.length = length
        self.num_all_entities = constants.NUM_ALL_ENTITIES
        self.num_entities_task = NUM_ENTITIES[task]

        # self.grid_size = (STATE_HEIGHT, STATE_WIDTH)
        self.grid_size = size
        self.vis = vis
        self.mode = mode

        if self.fname().exists():
            self._load_mean_sent_embeds()
        else:
            raise ValueError("No embeddings found", self.fname())

    # def apply_augmentation(self, entity_pos, avatar_pos):
    #     # Apply the chosen augmentation
    #     return self.aug_method2(*self.aug_method1(entity_pos, avatar_pos))

    # s1_train_sent_embeddings
    def fname(self):
        if self.t5_sent:
            return _data_dir() / f"{self.task}_{self.mode}_sent_embeddings.pkl"
        else:
            if "lwm" in self.task:
                mode = "train_eval_test"
                task = "lwm"
                return _data_dir() / "messenger" / f"{mode}_{task}_{self.model_sent}.pkl"

            # if self.test_sent_emb_only:
            #     if self.mode == "eval":
            #         mode = "train_eval_test"
            #     elif "test" in self.mode:
            #         mode = "test"
            #     else:
            #         raise ValueError(f"{self.mode=}")
            # else:
            mode = "train_eval_test"
            return (
                _data_dir()
                / "messenger"
                / f"{mode}_{self.task}_{self.model_sent}.pkl"
            )

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
        # if not self.save_sent_emb:
        #     del self.id2sentemb

    def _update_pos_entity_avatar(self, obs):
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

    # def _get_ids_pos_entity_avatar(self, obs):
    #     entity_pos = np.nonzero(obs["entities"])  # (Ne), (Ne), (Ne)
    #     entity_ids = obs["entities"][entity_pos]  # (Ne)
    #     avatar_pos = np.nonzero(obs["avatar"])  # ([1]), ([1]), ([1])
    #     avatar_ids = obs["avatar"][avatar_pos]  # (1)
    #     entity_pos = np.stack(
    #         [entity_pos[0], entity_pos[1], entity_ids], axis=-1
    #     )  # (Ne, 3)
    #     num_entities_current = entity_pos.shape[0]

    #     if num_entities_current < self.num_entities_task:
    #         # add death
    #         dealth_id = np.ones((self.num_entities_task - num_entities_current)) * DEATH_ID  # (N)
    #         # add dealth_ids as the last dim of dealth_pos
    #         dealth_pos = (
    #             np.ones((self.num_entities_task - num_entities_current, 2)) * DEATH_POS
    #         )  # (N, 2 )
    #         dealth_pos = np.concatenate(
    #             [dealth_pos, dealth_id[:, None]], axis=-1
    #         )  # (N, 3)
    #         entity_pos = np.concatenate([entity_pos, dealth_pos])  # (N, 3)
    #         entity_ids = np.concatenate([entity_ids, dealth_id])  # N, 3

    #     if avatar_ids.shape[0] == 0:
    #         avatar_ids = np.array([DEATH_ID])
    #         avatar_pos = (np.array([DEATH_POS]), np.array([DEATH_POS]))

    #     assert np.all(entity_ids < self.num_all_entities), f"{entity_ids=} {self.num_all_entities=}"
    #     # assert np.all(entity_pos < 10), f"{entity_pos=} {self.grid_size=}"
    #     avatar_pos = np.stack(
    #         [avatar_pos[0], avatar_pos[1], avatar_ids], axis=-1
    #     )  # (1, 3)
    #     # assert obs["image"][avatar_pos] == avatar_ids[0]

    #     if self.aug:
    #         entity_pos, avatar_pos = self.apply_augmentation(entity_pos, avatar_pos)

    #     return {
    #         "entity_ids": entity_ids.astype(int),
    #         "entity_pos": entity_pos.astype(int),
    #         "avatar_ids": avatar_ids.astype(int),
    #         "avatar_pos": avatar_pos.astype(int),
    #     }

    def _get_entity_tracking(self, obs, reset=False):  # hist_len, N, 3
        """
        get entity_pos_hist and avatar_pos_hist
        """
        entity_pos = np.copy(obs["entity_pos"])[:, :2]
        avatar_pos = np.copy(obs["avatar_pos"])[:, :2]
        history_step = min(self._step, self.hist_len - 1)
        if reset:
            assert self._step == 0, self._step
            # self.entity_pos_hist.clear()
            # self.avatar_pos_hist.clear()
            # self.entity_pos_hist = np.zeros((self.hist_len, self.num_entities_task, 2))
            # self.avatar_pos_hist = np.zeros((self.hist_len, 1, 2))
            # fill with -1
            self.entity_pos_hist = np.ones(
                (self.hist_len, self.num_entities_task, 2)
            ) * (-1)
            self.avatar_pos_hist = np.ones((self.hist_len, 1, 2)) * (-1)
            self.entity_vel_hist = np.zeros((self.hist_len, self.num_entities_task, 2))

            self.entity_pos_hist[0] = entity_pos
            self.avatar_pos_hist[0] = avatar_pos

            # fill with initial positions
            # for _ in range(self.hist_len):
            # self.entity_pos_hist.append(entity_pos)
            # self.avatar_pos_hist.append(avatar_pos)
            # self.entity_vel_hist.append(np.zeros_like(entity_pos))

        else:
            assert self._step > 0, self._step

            if self._step < self.hist_len:
                previous_entity_pos = self.entity_pos_hist[history_step - 1]
                previous_avatar_pos = self.avatar_pos_hist[history_step - 1]
            else:
                previous_entity_pos = self.entity_pos_hist[-1].copy()
                previous_avatar_pos = self.avatar_pos_hist[-1].copy()
                self.entity_pos_hist[:-1] = self.entity_pos_hist[1:]
                self.avatar_pos_hist[:-1] = self.avatar_pos_hist[1:]
                self.entity_vel_hist[:-1] = self.entity_vel_hist[1:]

            if DEAD_POS in entity_pos:
                for i in range(self.num_entities_task):
                    if DEAD_POS in entity_pos[i]:
                        entity_pos[i] = previous_entity_pos[i]

            self.entity_vel_hist[history_step] = entity_pos - previous_entity_pos
            self.entity_pos_hist[history_step] = entity_pos

            if DEAD_POS in avatar_pos:
                avatar_pos = previous_avatar_pos
            self.avatar_pos_hist[history_step] = avatar_pos

        pos_info = {}
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
        pos_info = {
            k: v.transpose(1, 0, 2).reshape(v.shape[1], -1) for k, v in pos_info.items()
        }

        pos_vec_hist = np.array(self.entity_pos_hist) - np.array(self.avatar_pos_hist)
        entity_vec_hist = np.array(self.entity_vel_hist)
        norm = np.linalg.norm(pos_vec_hist, axis=-1, keepdims=True)
        # Replace small values with 1 to avoid division by zero
        norm = np.where(norm < EPSILON, 1, norm)
        normed_pos_vec_hist = pos_vec_hist / norm

        assert not np.isnan(pos_vec_hist).any(), (
            f"NaN detected in pos_vec_hist: {pos_vec_hist}"
        )
        assert not np.isnan(entity_vec_hist).any(), (
            f"NaN detected in entity_vec_hist: {entity_vec_hist}"
        )
        assert not np.isnan(norm).any(), f"NaN detected in norm: {norm}"

        # assert norm doesn't have nan
        assert not np.isnan(normed_pos_vec_hist).any(), f"{normed_pos_vec_hist=}"
        pos_vec_hist = normed_pos_vec_hist.astype(np.float32)  # (hist_len, N, 2)
        # N, hist_len
        pos_info["dp"] = np.sum(pos_vec_hist * entity_vec_hist, axis=-1).T

        # mask out -1 for future steps
        if self.mask_future_steps_dp:
            pos_info["dp"][:, history_step + 1 :] = -1

        return pos_info

    def _get_movement_class(self, obs):
        res = np.zeros((self.num_entities_task))
        assert self.movement_classes is not None, self.movement_classes
        for i in range(self.num_entities_task):
            res[i] = MOVEMENT_2_ID[self.movement_classes[i]]
        return res.astype(np.int32)

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
        # assert (
        #     new_ob.shape == self.observation_space["image"].shape
        # ), f'{new_ob.shape=} {self.observation_space["image"].shape=}'
        return new_ob

    @property
    def observation_space(self):
        # from messenger.envs.config import STATE_HEIGHT, STATE_WIDTH
        # original_grid_size = (STATE_HEIGHT, STATE_WIDTH)
        obs_space = {}
        # obs_space["game_id"] = spaces.Box(
        #     low=0, high=MAX_GAMES[self.task], shape=(), dtype=int
        # )

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
            # obs_space["entity_pos_hist"] = spaces.Box(
            #     low=-1,
            #     high=10,  # max(16, 16, 17)
            #     # shape=(HIST_LEN, self.num_entities_task, 2),
            #     shape=(self.num_entities_task, 2 * HIST_LEN),
            # )
            # obs_space["avatar_pos_hist"] = spaces.Box(
            #     low=-1,
            #     high=10,
            #     # shape=(HIST_LEN, 1, 2),
            #     shape=(1, 2 * HIST_LEN),
            # )
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
            obs_space["movement"] = spaces.Box(
                low=0,
                high=3,
                shape=(self.num_entities_task,),
            )

            obs_space["dp"] = spaces.Box(
                low=-1,
                high=3,
                shape=(self.num_entities_task, self.hist_len),
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

        # if self.vis:
        #     assert not self.remove_image
        #     obs_space["log_image"] = spaces.Box(
        #         low=0,
        #         high=255,
        #         shape=(100 * self.grid_size[0], 100 * self.grid_size[1], 3),
        #     )

        # if self.use_read:
        #     obs_space["is_read_step"] = spaces.Box(
        #         low=np.array(False),
        #         high=np.array(True),
        #         shape=(),
        #         dtype=bool,
        #     )
        return spaces.Dict(obs_space)

    @property
    def action_space(self):
        return self._env.action_space

    # def _embed(self, string):
    #     if not hasattr(self, "token_cache"):
    #         self._load_token_embeds()

    #     if (
    #         f"{string}_{self.max_token_seqlen}" not in self.token_cache
    #         or string not in self.embed_cache
    #     ):
    #         print(f"not in token cache: {string=} ")
    #         tokens = self.tokenizer(
    #             string, return_tensors="pt", add_special_tokens=True
    #         )  # add </s> separators

    #         import torch

    #         with torch.no_grad():
    #             # (seq, dim)
    #             assert isinstance(self.encoder, T5EncoderModel)
    #             embeds = self.encoder(**tokens).last_hidden_state.squeeze(0)

    #         self.embed_cache[string] = embeds.cpu().numpy()
    #         self.token_cache[f"{string}_{self.max_token_seqlen}"] = {
    #             k: v.squeeze(0).cpu().numpy() for k, v in tokens.items()
    #         }
    #         self.check_to_save_emb_cache()

    #     return (
    #         self.embed_cache[string],
    #         self.token_cache[f"{string}_{self.max_token_seqlen}"],
    #     )

    # def _get_mean_sent_from_tokens(self):
    #     es, ts = self._embed(sent)
    #     mean_sent = es[ts["attention_mask"].astype(bool)].mean(0)
    #     return mean_sent

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

    def create_game_config(self):
        return self._env.create_game_config()

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
        self.manual = "</s>" + "</s>".join([x.strip() for x in self.manual_sentences])
        # to feed gt role to wm
        if self.use_movement_class:
            from messenger.envs.stage_one import StageOne

            if isinstance(self._env, TwoEnvWrapper):
                assert self._env.cur_env is not None, self._env.cur_env
                assert not isinstance(self._env.cur_env, StageOne)

                self.movement_classes = self._env.cur_env.movement_classes
            else:
                assert not isinstance(self._env, StageOne)
                self.movement_classes = self._env.movement_classes

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
        if self.use_movement_class:
            self.movement = self._get_movement_class(obs)
            obs["movement"] = self.movement

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

        # obs["is_read_step"] = self.reading
        if "entities" in obs:
            del obs["entities"]
        if "avatar" in obs:
            del obs["avatar"]

        # if self.use_read:
        #     self.reading = True
        #     obs["is_read_step"] = self.reading
        #     self.read_step = 1
        #     if self.read_step >= self.max_read_steps:
        #         self.reading = False

        self._init_obs = obs
        return obs  # no reward here

    def step(self, action):
        # if self.aug:
        #     # ACTIONS = namedtuple("Actions", "up down left right stay")(0, 1, 2, 3, 4)
        #     if self.aug_method1.__name__ == "flip_horizontally":
        #         # reverse action for flip - swap (left, right)
        #         if action == 2:
        #             action = 3
        #         elif action == 3:
        #             action = 2
        #     elif self.aug_method1.__name__ == "flip_vertically":
        #         # reverse action for flip - swap (up, down)
        #         if action == 0:
        #             action = 1
        #         elif action == 1:
        #             action = 0
        #     elif self.aug_method1.__name__ == "flip_both":
        #         # reverse action for flip - swap (up, down) and (left, right)
        #         if action == 0:
        #             action = 1
        #         elif action == 1:
        #             action = 0
        #         elif action == 2:
        #             action = 3
        #         elif action == 3:
        #             action = 2

        # print("action=", action)
        # if self.use_read and self.reading:
        #     assert self.read_step <= self.max_read_steps, self.read_step
        #     assert self._init_obs is not None, self._init_obs
        #     obs = self._init_obs
        #     assert obs is not None, obs
        #     self.read_step += 1
        #     if self.read_step >= self.max_read_steps:
        #         self.reading = False
        #         # self.read_step = 0
        #     return obs, 0, False, {}

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
        if self.use_movement_class:
            obs["movement"] = self.movement

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

        # if self.use_read:
        #     obs["is_read_step"] = self.reading

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


NUM_TIME_STEPS = {
    "s1": 5,
    "s2": 33,
    "s3": 65,
    "easy": 33,
    "medium": 33,
    "hard": 33,
    "lwm": 33,
}
ACTIONS = namedtuple("Actions", "up down left right stay")(0, 1, 2, 3, 4)


def test_history_pos():
    _add_checkout_paths()
    env = MessengerSent(
        "s2", mode="train", length=32, use_movement_class=True, entity_track=True
    )
    for _ in tqdm(range(30)):
        obs = env.reset()
        # print(env.entity2manual)
        print(env.manual)
        print(env.movement_classes)

        # check obs['movement'] doesn't change
        movements = [obs["movement"]]
        for entity_id, move_class, manual_id in zip(
            obs["entity_ids"], obs["movement"], obs["manual_ids"]
        ):
            entity_name = IDX_TO_ENTITY_NAME[entity_id]
            move_class = ID_2_MOVEMENT[move_class]
            sent = env.manual_sentences[manual_id]
            print(f"{entity_name=}, {move_class=}, {sent=}")

        print("")
        print(obs)

        for t in range(32):
            action = random.choice([0, 1, 2, 3, 4])
            obs, reward, done, info = env.step(action)
            movements.append(obs["movement"])
            # print(f"{t=}")
            print(obs)
            # print(reward)
            # print("")
            SMALL_ENOUGH_TIME = 3

            # assert movement_class is correct based on dp
            for e_id in range(env.num_entities_task):
                e_dp = obs["dp"][e_id]
                e_move_id = obs["movement"][e_id]
                e_move_class = ID_2_MOVEMENT[e_move_id]
                # MOVEMENT2ID = {"immovable": 0, "fleeing": 1, "chaser": 2}

                if e_move_id == 0:
                    # assert e_dp is full of 0
                    assert np.all(e_dp == 0), f"{e_dp=}, {e_move_id=}, {e_id=}"

                elif e_move_id == 1:
                    # assert e_dp is most of the time <= 0, can be positive but a few times

                    assert np.sum(e_dp > 0) < SMALL_ENOUGH_TIME, (
                        f"{e_dp=}, {e_move_class=}, {e_id=}"
                    )

                elif e_move_id == 2:
                    assert np.sum(e_dp < 0) < SMALL_ENOUGH_TIME, (
                        f"{e_dp=}, {e_move_class=}, {e_id=}"
                    )

                else:
                    raise ValueError(f"{e_move_class=}")

            if done:
                break

        movements = [tuple(movement) for movement in movements]
        assert len(set(movements)) == 1, movements

    cprint("PASS!!!", "green")
    # same for obs['entity_ids']


def test_deter_game():
    _add_checkout_paths()
    env = MessengerSent("s2", mode="train", length=32, deter_game=True)
    NUM_GAMES = 100

    for _ in tqdm(range(NUM_GAMES), "games"):
        # obs = env.reset()
        game_config = env.create_game_config()
        # make sure it runs randomly everytime
        actions_root = [random.choice([0, 1, 2, 3, 4]) for _ in range(32)]
        # print(actions_root)
        rewards_root = np.ones(32) * (-1)
        dones_root = np.zeros(32)
        env.reset_game_config(**game_config)
        obs = env.reset()

        for t in range(32):
            obs, reward, done, info = env.step(actions_root[t])
            rewards_root[t] = reward
            dones_root[t] = done
            if done:
                break

        # print(rewards_root)
        # print(dones_root)
        # assert dones_root has 1
        assert np.any(dones_root), "There should be at least one done"

        NUM_ENVS = 200
        for _ in tqdm(range(NUM_ENVS), "run duplicated envs"):
            env.reset()
            rewards = np.ones(32) * (-1)
            dones = np.zeros(32)
            for t in range(32):
                obs, reward, done, info = env.step(actions_root[t])
                rewards[t] = reward
                dones[t] = done
                if done:
                    break

            assert np.all(rewards == rewards_root), (rewards, rewards_root)
            assert np.all(dones == dones_root), (dones, dones_root)

    cprint("PASS", "green")


if __name__ == "__main__":
    # test_history_pos()
    test_deter_game()
