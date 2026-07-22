# %%
from collections import defaultdict
import json
from typing import List

games_path = "./custom_text_splits/data_splits_final_with_messenger_names.json"
games = json.load(open(games_path, "r"))

# %%

LWM_SPLIT_MAP = {
    "easy": "ne_sr_and_sm",
    "medium": "se_nr_or_nm",
    "hard": "ne_nr_or_nm",
}
print(games.keys())
test_games = games["test_" + LWM_SPLIT_MAP["hard"]]
# print(test_games)


# %%
def move_role_games(games):
    res = []
    for game in test_games:
        entities_game = sorted(((e[1], e[2]) for e in game))
        res.append(entities_game)
    return res


test_mr_games = move_role_games(test_games)
train_games = games["train"]
train_mr_games = move_role_games(train_games)


def convert_to_sets(games: List[List[List[str]]]):
    res = []
    for game in games:
        res.append(tuple(sorted(tuple(e) for e in game)))
    return set(res)


test_mr_sets = convert_to_sets(test_mr_games)
train_mr_sets = convert_to_sets(train_mr_games)

# %%
# count frequency of each game
from collections import Counter

# print(test_mr_games)
# test_mr_games = [tuple(e) for e in test_mr_games]
# print(Counter(test_mr_games))

train_mr_games = [tuple(e) for e in train_mr_games]
print(Counter(train_mr_games))
print(len(train_mr_games))


# %%
print(len(test_mr_games))
print(len(train_mr_games))
print(len(test_mr_sets))
print(len(train_mr_sets))

# %%
print(train_mr_sets)
print("")

print(test_mr_sets)
# %%
# find intersection of train_mr_sets and test_mr_sets
print(len(test_mr_sets.intersection(train_mr_sets)))
