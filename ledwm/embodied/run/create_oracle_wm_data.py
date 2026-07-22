from collections import Counter
import json
import os
import random
import numpy as np
from messenger.envs.config import ENTITY_IDS
from termcolor import cprint
from tqdm import tqdm
from ledwm.embodied.core.OracleAgent import OracleAgent, INTENTIONS
from ledwm.embodied.core.basics import convert
from ledwm.embodied.envs.from_gym import FromGym
from ledwm.embodied.run.smoothing import ReplayEps
from collections import defaultdict


def _make_int_defaultdict():
    """Picklable function to create defaultdict(int)"""
    return defaultdict(int)


class DataGenerator:
    def __init__(
        self,
        env: FromGym,
        replay: ReplayEps,
        logger,
        config,
        seed=1000,
        num_repeat=300,
        behavior_policy="mixed",
        split="train",
    ):
        # load data
        emma_folder = "messenger-emma/messenger/envs/texts/custom_text_splits"
        with open(
            os.path.join(emma_folder, "data_splits_final_with_messenger_names.json")
        ) as f:
            self.splits = json.load(f)

        with open(
            os.path.join(
                emma_folder,
                "custom_text_splits_with_messenger_names.json",
            )
        ) as f:
            self.texts = json.load(f)

        SPLIT_MAP = {
            "easy": "test_ne_sr_and_sm",
            "medium": "test_se_nr_or_nm",
            "hard": "test_ne_nr_or_nm",
        }

        if split == "train":
            self.splits = self.splits["train"]

        elif split in ["easy", "medium", "hard"]:
            self.splits = self.splits[SPLIT_MAP[split]]
            assert behavior_policy == "test", f"{behavior_policy=}"

        else:
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        self.num_repeats = num_repeat
        self.behavior_policy = behavior_policy
        self.is_train = behavior_policy == "mixed"

        self.random = random.Random(seed + 2340)
        self.policy = OracleAgent(seed=seed)
        self.env = env
        self.replay = replay
        self.logger = logger
        self.config = config

    def generate_data(self, movement_role_config_allowed=None):
        reward_collected_all = Counter()

        if self.config.movement_role_lang_grid:
            # set ((movement1, role1), (movement2, role2), (movement3, role3))
            self.movement_role_configs = defaultdict(_make_int_defaultdict)

        for game in tqdm(self.splits, desc=f"Generating {self.split} data"):
            if not self.env.has_manual_given_game(game):
                cprint(
                    f"Skipping {game} because it has no manual in {self.split}",
                    "yellow",
                )
                continue

            game_config = self.env.create_game_config(entities=game)

            if self.config.offline_one_config:
                if not self.is_train:
                    assert movement_role_config_allowed is not None, (
                        "movement_role_config_allowed must be provided for test"
                    )
                    movement_role_config = tuple(
                        sorted([(e[1], e[2]) for e in game_config["entities"]])
                    )
                    init_state = game_config["init_state"]
                    if movement_role_config not in movement_role_config_allowed:
                        continue

            if self.is_train and self.config.movement_role_lang_grid:
                found_all_eps = False

                while True:
                    movement_role_config = tuple(
                        sorted([(e[1], e[2]) for e in game_config["entities"]])
                    )
                    init_state = game_config["init_state"]

                    if self.config.offline_one_config:
                        if len(self.movement_role_configs) == 1:
                            if movement_role_config not in self.movement_role_configs:
                                found_all_eps = True
                                break

                    if (
                        self.movement_role_configs[movement_role_config][init_state]
                        == 0
                    ):
                        cprint(
                            f"Found new config {movement_role_config} {init_state}",
                        )
                        break

                    if len(self.movement_role_configs[movement_role_config]) >= 24:
                        cprint("Find all eps for this config. COntinue games", "red")

                        found_all_eps = True
                        break

                    if (
                        self.movement_role_configs[movement_role_config][init_state]
                        >= 0
                    ):
                        cprint(
                            f"Skipping {movement_role_config} {init_state} because it has already been seen",
                            "yellow",
                        )
                        game_config = self.env.create_game_config(entities=game)

                if found_all_eps:
                    continue
                else:
                    self.movement_role_configs[movement_role_config][init_state] += 1

            self.env.reset_game_config(**game_config)

            reward_eps_collected = Counter()
            for n in range(self.num_repeats):
                if self.behavior_policy == "mixed":
                    episode_intention = self.random.choice(INTENTIONS)

                elif self.behavior_policy == "test":
                    assert self.num_repeats == len(INTENTIONS), (
                        f"Number of repeats {self.num_repeats} must be equal to the number of intentions {len(INTENTIONS)}"
                    )
                    episode_intention = INTENTIONS[n]

                else:
                    assert self.behavior_policy in INTENTIONS, (
                        f"Behavior policy {self.behavior_policy} not in {INTENTIONS}"
                    )

                rollout_result = self.rollout(game, episode_intention, game_config)
                if rollout_result is None:
                    break

                reward_eps_collected[rollout_result] += 1

            reward_collected_all += reward_eps_collected
            print(f"{reward_eps_collected=}")
            print(f"{reward_collected_all=}")
            print("")

        self.replay.save()

    def get_action(self, obs, true_parsed_manual, episode_intention):
        action = self.policy.act(obs, true_parsed_manual, episode_intention)
        assert action is not None
        assert type(action) is int, "action must be an integer"
        # turn 1 hot action into one hot action
        num_actions = self.env.act_space["action"].shape[0]
        action = {
            "action": np.eye(num_actions)[action],
            "reset": False,
        }
        assert action["action"].shape == (num_actions,), (
            f"Action shape {action['action'].shape} not equal to {num_actions}"
        )
        return action

    def rollout(self, game, episode_intention, game_config):
        reset_action = {
            k: convert(np.zeros(v.shape)) for k, v in self.env.act_space.items()
        }
        reset_action["reset"] = True  # type: ignore
        obs = self.env.step(reset_action)

        next_action = self.get_action(
            obs, self.env._env.true_parsed_manual, episode_intention
        )
        trans = {**obs, **next_action}

        if self.is_train and self.config.movement_role_lang_grid:
            # print("role=", obs["role"])
            assert not all(x == 0 for x in obs["role"]), f"{obs=}"

        self.replay.add(trans)

        true_parsed_manual = self.env._env.true_parsed_manual
        assert game == true_parsed_manual, (
            f"Game {game} not equal to {true_parsed_manual}"
        )
        for entity_info in true_parsed_manual:
            entity_id = ENTITY_IDS[entity_info[0]]
            assert entity_id in obs["entity_ids"], (
                f"Entity {entity_id} not in {obs['entity_ids']}"
            )

        # run until done
        reward_eps_sum = 0
        has_message = False
        while True:
            obs = self.env.step(next_action)
            has_message = obs["avatar_ids"][0] == 16

            if obs["is_last"]:
                next_action = self.get_action(
                    obs, self.env._env.true_parsed_manual, "random"
                )

                # check if entity or avatar disappear
                if self.config.env.lwm.disappear:
                    pass
                    # num_alive_entities = [e for e in obs["entity_ids"] if e != 0]
                    # is_dead_agent = obs["avatar_ids"][0] == 0
                    # assert is_dead_agent or len(num_alive_entities) < 3, (
                    #     obs["entity_ids"],
                    #     obs["avatar_ids"],
                    # )

                else:
                    num_alive_entities = [e for e in obs["entity_ids"] if e != 0]
                    if obs["reward"] == -1:
                        if has_message:
                            assert len(num_alive_entities) == 2, f"{obs['entity_ids']=}"
                        else:
                            assert len(num_alive_entities) == 3, f"{obs['entity_ids']=}"

                    if obs["reward"] == 1:
                        assert len(num_alive_entities) == 2, f"{obs['entity_ids']=}"
                    assert obs["avatar_ids"][0] in [15, 16], f"{obs['avatar_ids']=}"
            else:
                next_action = self.get_action(
                    obs, self.env._env.true_parsed_manual, episode_intention
                )

            trans = {**obs, **next_action}
            self.replay.add(trans)
            reward_eps_sum += obs["reward"]
            if obs["is_last"]:
                break

        return reward_eps_sum
