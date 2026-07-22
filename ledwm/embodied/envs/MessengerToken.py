import os
import pickle

import embodied
import numpy as np
from gym import spaces
from termcolor import cprint
from transformers import T5EncoderModel, T5Tokenizer

NUM_SENTS = {
    "s1": 3,
    "s2": 3,
    "s3": 6,
}

NUM_ENTITIES = {
    "s1": 3,
    "s2": 3,
    "s3": 5,
}

NUM_CHANNELS = {"s1": 2}
UNK_TOKEN_ID = 2
PAD_TOKEN_ID = 0
UNK = "<unk>"
PAD = "<pad>"


class MessengerToken(embodied.Env):
    def __init__(
        self,
        task,
        mode="train",
        size=(16, 16),
        length=64,
        load_embeddings=True,
        vis=False,
        small=False,
        small_image=False,
        small_image_decoder=False,
        lang=True,
        use_reading=True,
        use_sent_ids=True,
        model_sent="T5",
        debug_pos_gt=False,
        use_time_step=True,
        remove_image=True,
    ):
        cprint(f"Messenger config: {task=}, {mode=}, {length=}", "green")
        assert task in ("s1", "s2", "s3")
        assert mode in ("train", "eval", "test")
        self.small_image = small_image
        self.small_image_decoder = small_image_decoder
        self.lang = lang
        self.use_reading = use_reading
        self.use_sent_ids = use_sent_ids
        self.num_all_entities = 17
        self.num_entities_task = NUM_ENTITIES[task]
        self.remove_image = remove_image
        self.num_sents = NUM_SENTS[task]
        # self.use_time_step = use_time_step

        from messenger.envs.stage_one import StageOne
        from messenger.envs.stage_three import StageThree
        from messenger.envs.stage_two import StageTwo
        from messenger.envs.TwoEnvWrapper import TwoEnvWrapper

        # from messenger.envs.config import STATE_HEIGHT, STATE_WIDTH
        from . import from_gym

        self.task = task
        if task == "s1":
            if mode == "train":
                self._env = TwoEnvWrapper(
                    stage=1,
                    split_1="train-mc",
                    split_2="train-sc",
                    prob_env_1=0.75,
                    small=small,
                )
            elif mode == "eval":
                self._env = StageOne(split="val")
            else:
                assert mode == "test"
                self._env = StageOne(split="test")

        elif task == "s2":
            if mode == "train":
                self._env = TwoEnvWrapper(
                    stage=2, split_1="train-sc", split_2="train-mc"
                )
            elif mode == "eval":
                self._env = StageTwo(split="val")
            else:
                assert mode == "test"
                self._env = StageTwo(split="test")

        elif task == "s3":
            if mode == "train":
                self._env = TwoEnvWrapper(
                    stage=3,
                    split_1="train-mc",
                    split_2="train-sc",
                    prob_env_1=0.75,
                )
            elif mode == "eval":
                self._env = StageThree(split="val")
            else:
                assert mode == "test"
                self._env = StageThree(split="test")

        # Wrappers
        self.wrappers = [from_gym.FromGym]
        if not self.remove_image:
            self.wrappers.append(
                # Pad image to multiple of 2
                lambda e: wrappers.PadImage(
                    e, "small_image" if self.small_image else "image", size
                )
            )

        self.max_token_seqlen = 36  # per manual sentence
        self.manual = None
        self.tokens = []
        self.current_sentence = None
        self._step = 0
        self._init_obs = None
        self.length = length
        self.is_reading = False
        self.read_step = 0

        self.n_entities = 17
        # self.grid_size = (STATE_HEIGHT, STATE_WIDTH)
        self.grid_size = size
        print(f"Resize image to {self.grid_size=}")

        self.vis = vis
        self.mode = mode

        if mode == "test":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.encoder = T5EncoderModel.from_pretrained("t5-small")

        # if load_embeddings:
        fname = self.fname()
        with open(fname, "rb") as f:
            self.token_cache, self.embed_cache = pickle.load(f)
            print(
                f"Loading {len(self.token_cache)} embedding sents from {fname}",
                # "green",
            )
        self.pad_token_id = PAD_TOKEN_ID
        self.pad_token_embed = self.embed_cache[PAD][0]

        if not self.lang:
            self.read_token_id = UNK_TOKEN_ID
            self.read_token_embed = self.embed_cache[UNK][0]

        # else:
        #     self._init_models()

    # def _init_models(self):
    #     self.token_cache = {}  # {**strings**: {input_ids, attention_mask}
    #     self.embed_cache = {}

    #     # if self.tokenizer and self.encoder is not initialized, then initialize them
    #     if getattr(self, "tokenizer", None) is None:
    #         self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
    #     if getattr(self, "encoder", None) is None:
    #         self.encoder = T5EncoderModel.from_pretrained("t5-small")

    #     self.pad_token_id = self.tokenizer.pad_token_id
    #     self.pad_token_embed = self._embed(PAD)[0][0]
    #     if not self.lang:
    #         self.read_token_embed = self._embed(UNK)[0][0]
    #         # assert self.read_token_embed is different from empty_embed, they are np array
    #         # and not the same value
    #         # TODO
    #         self.read_token_id = self.tokenizer.encode(UNK)[0]
    #         assert self.read_token_id != self.pad_token_id

    def _symbolic_to_multihot(self, obs):
        # (h, w, 2)
        layers = np.concatenate((obs["entities"], obs["avatar"]), axis=-1).astype(int)
        new_ob = np.maximum.reduce(
            [np.eye(self.n_entities)[layers[..., i]] for i in range(layers.shape[-1])]
        )
        new_ob[:, :, 0] = 0
        # assert (
        #     new_ob.shape == self.observation_space["image"].shape
        # ), f'{new_ob.shape=} {self.observation_space["image"].shape=}'
        return new_ob

    @property
    def observation_space(self):
        obs_space = {
            "image": spaces.Box(
                low=0,
                high=1,
                shape=(*self.grid_size, self.n_entities),
            ),
            "is_read_step": spaces.Box(
                low=np.array(False),
                high=np.array(True),
                shape=(),
                dtype=bool,
            ),
        }
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
        # if self.small_image:
        #     obs_space["small_image"] = spaces.Box(
        #         low=0,
        #         high=1,
        #         shape=(*self.grid_size, NUM_CHANNELS[self.task]),
        #     )

        if self.use_reading:
            obs_space.update(
                {
                    "token": spaces.Box(0, 32100, shape=(), dtype=np.uint32),
                    "token_embed": spaces.Box(
                        -np.inf, np.inf, shape=(512,), dtype=np.float32
                    ),
                }
            )

        # if self.vis:
        #     obs_space["log_image"] = spaces.Box(
        #         low=0,
        #         high=255,
        #         shape=(100 * self.grid_size[0], 100 * self.grid_size[1], 3),
        #     )
        return spaces.Dict(obs_space)  # type: ignore

    @property
    def action_space(self):
        return self._env.action_space

    def _embed(self, string):
        assert string in self.embed_cache, f"{string=} {self.mode=}, {self.task=}"
        return (
            self.embed_cache[string],
            self.token_cache[f"{string}_{self.max_token_seqlen}"],
        )

    def fname(self):
        return f"{os.path.dirname(__file__)}/data/messenger/messenger_token_{self.task}_{self.mode}.pkl"

    # @property
    # def env_cache(self):
    #     if self.use_sent_ids:
    #         assert len(self.sent2id) == len(
    #             self.id2sentemb
    #         ), f"{len(self.sent2id)=} {len(self.id2sentemb)=}"
    #         # turn self.mean_sent_cache: list of np.array -> np.array
    #         return {
    #             "sent_embed": np.array(self.id2sentemb),
    #             "sent_ids": self.sent2id,
    #         }
    #     else:
    #         return None

    def reset(self):
        self._step = 0
        self.read_step = 0
        obs, self.manual_sentences = self._env.reset()
        obs: dict
        self.manual = "</s>" + "</s>".join([x.strip() for x in self.manual_sentences])

        if self.use_reading:
            self.token_embeds = []
            self.tokens = []  # for logging
            for sent in self.manual_sentences:
                es, ts = self._embed(sent)
                # ts: {'attention_mask', 'input_ids'}, es (l, dim) # Remove padding
                es = es[ts["attention_mask"].astype(bool)]
                ts = ts["input_ids"][ts["attention_mask"].astype(bool)]
                self.token_embeds += [tok_e for tok_e in es]
                self.tokens += [tok for tok in ts]

            assert len(self.token_embeds) == len(self.tokens), (
                f"{len(self.token_embeds)=} {len(self.tokens)=}"
            )
            if self.lang:
                obs.update(
                    {
                        "token": self.tokens[self.read_step],
                        "token_embed": self.token_embeds[self.read_step],
                    }
                )
            else:
                obs.update(
                    {
                        "token": self.read_token_id,
                        "token_embed": self.read_token_embed,
                    }
                )

            obs.update({"log_language_info": self.manual})
            self.is_reading = True
            self.read_step += 1
            obs["is_read_step"] = self.is_reading

        # if self.small_image:
        #     obs["small_image"] = np.concatenate(
        #         (obs["entities"], obs["avatar"]), axis=-1
        #     ).astype(int)
        # else:
        #     obs["image"] = self._symbolic_to_multihot(obs)

        obs.update(self._update_pos_entity_avatar(obs))
        # if self.vis:
        #     obs["log_image"] = self.make_image(obs["image"], -1, 0, False)

        if "entities" in obs:
            del obs["entities"]
        if "avatar" in obs:
            del obs["avatar"]
        self._init_obs = obs
        return obs

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

    def step(self, action):
        if self.use_reading:
            if self.is_reading:
                assert self._init_obs is not None, self._init_obs
                obs = self._init_obs
                obs["is_read_step"] = self.is_reading
                if self.lang:
                    obs["token"] = self.tokens[self.read_step]
                    obs["token_embed"] = self.token_embeds[self.read_step]
                else:
                    obs["token"] = self.read_token_id
                    obs["token_embed"] = self.read_token_embed

                obs.update(
                    {
                        "log_language_info": self.manual,
                        #        "log_tokens": self.tokens,
                    }
                )
                self.read_step += 1
                if self.read_step >= len(self.tokens):
                    self.is_reading = False
                    self.read_step = 0
                return obs, 0, False, {}

        self._step += 1  # don't increment step while reading
        obs, reward, done, info = self._env.step(action)
        obs: dict

        info = info or {}
        if self.use_reading:
            obs["is_read_step"] = self.is_reading
            obs["token"] = self.pad_token_id
            obs["token_embed"] = self.pad_token_embed
            obs.update(
                {
                    "log_language_info": self.manual,
                }
            )

        # if self.small_image:
        #     obs["small_image"] = np.concatenate(
        #         (obs["entities"], obs["avatar"]), axis=-1
        #     ).astype(int)
        # else:
        #     obs["image"] = self._symbolic_to_multihot(obs)
        # if self.vis:
        #     obs["log_image"] = self.make_image(obs["image"], action, reward, done)
        obs.update(self._update_pos_entity_avatar(obs))
        # info.update(
        #     {
        #         "entities": obs["entities"],
        #         "avatar": obs["avatar"],
        #     }
        # )
        if "entities" in obs:
            del obs["entities"]
        if "avatar" in obs:
            del obs["avatar"]

        # timeout
        if self._step >= self.length:
            done = True
            reward = -1
        return obs, reward, done, info

    # def make_image(self, img, ac, reward, done):
    #     assert len(img.shape) == 3
    #     assert img.shape[2] == 17
    #     # Remove padding
    #     img = img[:10, :10]

    #     idx_to_letter = {
    #         2: "A",
    #         3: "M",
    #         4: "D",
    #         5: "B",
    #         6: "F",
    #         7: "C",
    #         8: "T",
    #         9: "H",
    #         10: "B",
    #         11: "R",
    #         12: "Q",
    #         13: "S",
    #         14: "W",
    #         15: "a",
    #         16: "m",
    #     }
    #     idx_to_entity_name = {
    #         2: "airplane",
    #         3: "mage",
    #         4: "dog",
    #         5: "bird",
    #         6: "fish",
    #         7: "scientist",
    #         8: "thief",
    #         9: "ship",
    #         10: "ball",
    #         11: "robot",
    #         12: "queen",
    #         13: "sword",
    #         14: "wall",
    #         15: "player",
    #         16: "player",
    #     }

    #     role_to_colors = {
    #         "player_with_message": "pink",
    #         "player_without_message": "orange",
    #         "message": "blue",
    #         "enemy": "red",
    #         "goal": "green",
    #         "other": "gray",
    #     }
    #     actions = ["up", "down", "left", "right", "stay", "reset"]
    #     scale = 256 / 10
    #     fontpath = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"
    #     font = ImageFont.truetype(fontpath, 12) if os.path.exists(fontpath) else None
    #     new_img = Image.new(size=(256, 256), mode="RGB", color=(31, 33, 50))
    #     draw = ImageDraw.Draw(new_img)
    #     idxs = img.argmax(-1)

    #     for i, row in enumerate(img):
    #         for j, col in enumerate(row):
    #             if idxs[i][j] == 0:
    #                 continue

    #             letter = idx_to_letter[idxs[i][j]]
    #             # x,y canvas reversed
    #             color = (247, 193, 119) if letter in ("a", "m") else (238, 108, 133)
    #             draw.text(
    #                 (int(j * scale), int(i * scale)), letter, fill=color, font=font
    #             )

    # #    manual = "//".join(self.manual_sentences)
    # #    manual = manual.encode("ascii", "ignore").decode("ascii")
    # #    chunk_size = 40
    # #    chunks = (len(manual) // chunk_size) + 1
    # #    manual2 = [manual[i * chunk_size:(i+1) * chunk_size] for i in range(0, chunks)]
    # #    manual2.append(f"a {actions[ac]} r {rewward} {done}")
    # #    draw.multiline_text((0, 0), "\n".join(manual2), (0, 0, 0))
    # new_img = np.asarray(new_img)
    # return new_img
