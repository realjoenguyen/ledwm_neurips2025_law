# %%
txt_path = "./text_train.json"
from collections import defaultdict
import json

# %%
data = json.load(open(txt_path, "r"))
new_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for entity, roles in data.items():
    for role, moves in roles.items():
        for move, move_strs in moves.items():
            if len(move_strs) > 0:
                print(entity, role, move)
                print(move_strs[0])
                new_data[entity][role][move] = [move_strs[0]]
            else:
                new_data[entity][role][move] = []


# %%
json.dump(new_data, open("./text_train_small.json", "w"))
