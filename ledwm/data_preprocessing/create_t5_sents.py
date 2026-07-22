# %%
import json

from termcolor import cprint

# TASK = "lwm"
TASK = "s1"
# TYPE = "test"
# TYPE = "train_eval"
TYPE = "train_eval_test"
# change gpu id in env variable
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def extract_sentences(json_data, keep_roles=None):
    sentences = []
    cprint(f"{keep_roles=}", "green")
    for role, category in json_data.items():
        for aspect in category.values():
            for state, content in aspect.items():
                if keep_roles is not None and state not in keep_roles:
                    # print(f"skip {state} in {role}")
                    continue
                if content:  # if the content list is not empty
                    sentences.extend(content)
    return sentences


def extract_sentences_from_file(file_path, keep_roles):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return extract_sentences(json_data, keep_roles)


def extract_sentences_from_lwm(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    # it has format like this
    #   "robot": {
    #     "chaser": {
    #       "message": {
    #         "train": [],
    #         "dev_se_nr_or_nm": [
    #           "you are approached by both the secret document and the bot.",
    #           "the humanoid that comes to you is a report that is classified.",
    #           "a bot is coming closer and has the message."
    #         ],
    sentences = []
    for entity, content1 in json_data.items():
        for movement, content2 in content1.items():
            for role, content3 in content2.items():
                for split in content3.keys():
                    sentences.extend(content3[split])
    return sentences


if TASK == "lwm":
    train_file_path = "messenger-emma/messenger/envs/texts/custom_text_splits/custom_text_splits_with_messenger_names.json"
    train_eval_test = extract_sentences_from_lwm(train_file_path)
    print(f"{len(train_eval_test)=}")
    print(f"{train_eval_test[:5]=}")

else:
    train_file_path = "messenger-emma/messenger/envs/texts/text_train.json"
    val_file_path = "messenger-emma/messenger/envs/texts/text_val.json"
    test_file_path = "messenger-emma/messenger/envs/texts/text_test.json"
    keep_roles = {"s1": ["immovable", "unknown"], "s2": None, "s3": None}[TASK]
    train_sentences = extract_sentences_from_file(train_file_path, keep_roles)
    eval_sentences = extract_sentences_from_file(val_file_path, keep_roles)
    test_sentences = extract_sentences_from_file(test_file_path, keep_roles)

    train_eval_sentences = train_sentences + eval_sentences
    # test_sentences
    train_eval_test = train_sentences + eval_sentences + test_sentences
    print(f"{len(train_sentences)=}")
    print(f"{len(eval_sentences)=}")
    print(f"{len(test_sentences)=}")


# %%

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_sentences_from_file():
    file = open(
        f"ledwm/embodied/envs/data/sentences_{TASK}_{TYPE}.txt",
        "r",
    )
    sents = []
    for line in file:
        sents.append(line.strip())
    file.close()
    return sents


from sentence_transformers import SentenceTransformer


# model = SentenceTransformer("sentence-transformers/sentence-t5-base")
# model_tag = "t5"

# model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# model_tag = 'mpnet'

model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
model_tag = "mini"

if TYPE == "train_eval":
    sents = train_eval_sentences
elif TYPE == "test":
    sents = test_sentences
elif TYPE == "train_eval_test":
    sents = train_eval_test

embs = model.encode(sents)
print("mean=", embs.mean())
print("std=", embs.std())
print("shape=", embs.shape)
res = {}
for i, sent in enumerate(sents):
    res[sent] = i

# %%
import pickle

path = f"ledwm/embodied/envs/data/messenger/{TYPE}_{TASK}_{model_tag}.pkl"
# if parent dir does not exist, create it
import os

if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
with open(path, "wb") as f:
    pickle.dump((embs, res), f)
print(f"Dumping {embs.shape} and {len(res)} to {path}")

# %%
ledwm_file_path = "ledwm/embodied/envs/data/messenger/messenger_embeds.pkl"

# tuple has 2 elements: info and embs
# key: sent + _36:
#   key (['input_ids', 'attention_mask'])
# embs: (num_token=36, dim=512)

import pickle

with open(ledwm_file_path, "rb") as f:
    data = pickle.load(f)

# print(len(data.keys()))
print(len(data))
sent = "the jet is chasing with the enemy."
print(data[0][sent + "_36"].keys())
print(data[1]["the jet is chasing with the enemy."].shape)
mask = data[0][sent + "_36"]["attention_mask"]
print(data[1]["the jet is chasing with the enemy."] * mask[:, None])
