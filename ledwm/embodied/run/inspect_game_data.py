# %%
games_path = "messenger-emma/messenger/envs/texts/custom_text_splits/data_splits_final_with_messenger_names.json"

import json

with open(games_path, "r") as f:
    games = json.load(f)

# print(games)
train_games = games["train"]
print(len(train_games))
# %%
from collections import Counter

move_role_counter = Counter()
for game in train_games:
    move_role_config = tuple(sorted([(e[1], e[2]) for e in game]))
    move_role_counter[move_role_config] += 1

print(move_role_counter)
# %%

# plot the move_role_counter
import matplotlib.pyplot as plt

# sort the move_role_counter by value
move_role_counter = dict(
    sorted(move_role_counter.items(), key=lambda x: x[1], reverse=True)
)
plt.bar(range(len(move_role_counter)), list(move_role_counter.values()))

# take the first leter of each tuple in the tuple key
plt.xticks(
    range(len(move_role_counter)),
    [f"({k[0][0]}{k[1][0]})" for k in move_role_counter.keys()],
    rotation=90,
)
plt.show()


# %%
