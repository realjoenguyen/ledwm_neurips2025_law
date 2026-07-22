# %%
path = "texts/custom_text_splits/data_splits_final_with_messenger_names.json"
# path = "texts/custom_text_splits/data_splits_downstream.json"
import json

with open(path, "r") as f:
    custom_text_splits = json.load(f)

# %%
print(custom_text_splits.keys())
len(custom_text_splits["test_ne_nr_or_nm"])
