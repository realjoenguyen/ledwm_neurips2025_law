# %%
import pathlib


OFFLINE_WM_ROOT = pathlib.Path("logdir/lwm/offline_wm")

train_dir = OFFLINE_WM_ROOT / "train" / "disappear=True" / "movement=True"
test_dir = OFFLINE_WM_ROOT / "test" / "disappear=True" / "movement=True" / "hard"
print(test_dir)
# logdir/lwm/offline_wm/test/disappear=True/movement=True/hard
# %%
# list all files in test_dir
test_files = list(test_dir.glob("*.npz"))
train_files = list(train_dir.glob("*.npz"))
# load each file
import numpy as np

from collections import Counter

# train_game_configs = set()  # [tuple(role, movement)]
from tqdm import tqdm


def is_empty_data(data):
    return np.all(data["entity_pos"] == 0)


def get_game_config(files):
    game_configs = []
    game_config_set = set()
    num_eps = 0

    print("there are ", len(files), " files")
    for file in tqdm(files, desc="Processing files"):
        # print(list(data.keys()))
        # print("")
        # print(data["role"][:50])
        # print(data["movement"][:50])
        data = np.load(file)
        # Validate that the actual data length matches the filename length
        # actual_lengths = {k: v.shape for k, v in data.items()}
        # print(f"{actual_lengths=}")

        print(data["role"].shape)
        for i in range(data["role"].shape[0]):
            if data["is_first"][i]:
                # train_game_configs.add((tuple(data["role"][i]), tuple(data["movement"][i])))
                assert max(data["movement"][i]) < 3, (
                    f"movement {data['movement'][i]} is out of range"
                )

                # if np.all(data["role"][i] == 0):
                #     print(f"num_eps: {num_eps}")
                #     print(file)
                #     print(i)
                #     print(data["entity_ids"][i : i + 10])
                #     print(data["entity_pos"][i : i + 10])
                #     print(data["role"][i : i + 10])
                #     print(data["movement"][i : i + 10])
                #     print(data["is_first"][i : i + 10])
                #     print(data["is_last"][i : i + 10])
                #     return

                if data["is_first"][i]:
                    if not np.all(data["role"][i] == 0):
                        num_eps += 1
                    else:
                        continue

                game_config = sorted(tuple(zip(data["role"][i], data["movement"][i])))
                # print(game_config)
                game_configs.append(tuple(game_config))
                game_config_set.add(tuple(game_config))
                # break
                if len(game_configs) % 1000 == 0:
                    print(Counter(game_configs))
        # break
        # print(train_game_configs)
        # print(len(game_configs))
        # if len(game_configs) == 27:
        #     break
        # if len(game_configs) % 100 == 0:
        #     print(Counter(game_configs))

    print(f"num_eps: {num_eps}")
    return game_configs, game_config_set


# %%
train_game_configs, train_game_config_set = get_game_config(train_files)
# test_game_configs, test_game_config_set = get_game_config(test_files)

print(len(train_game_config_set))
# print(len(test_game_config_set))

print(train_game_config_set)
# print(test_game_config_set)

# print counter of each game config
# from collections import Counter

# %%

# print()
# %%
print(Counter(train_game_configs))
# print(Counter(test_game_configs))

# %%
# %%

# %%
