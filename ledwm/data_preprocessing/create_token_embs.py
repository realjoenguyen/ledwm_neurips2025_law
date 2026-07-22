# %%
import sys
import gym
import messenger
from time import time
from termcolor import cprint
import torch
from tqdm import tqdm
import sys
from termcolor import cprint
import json

TASK = "s2"
TYPE = "train"
# TYPE = "test"
# TYPE = "eval"
# TYPE = "train_eval"
# TYPE = "train_eval_test"
import os

PAD = "<pad>"
UNK = "<unk>"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


train_file_path = "messenger-emma/messenger/envs/texts/text_train.json"
val_file_path = (
    "messenger-emma/messenger/envs/texts/text_val.json"
)
test_file_path = "messenger-emma/messenger/envs/texts/text_test.json"
keep_roles = {"s1": ["immovable", "unknown"], "s2": None, "s3": None}[TASK]
train_sentences = extract_sentences_from_file(train_file_path, keep_roles)
print(len(train_sentences))
print(train_sentences[:5])

if TYPE == "train":
    sentences = train_sentences
elif TYPE == "test":
    sentences = extract_sentences_from_file(test_file_path, keep_roles)
elif TYPE == "eval":
    sentences = extract_sentences_from_file(val_file_path, keep_roles)

assert len(sentences) > 0

# from ledwm.embodied.envs import MessengerToken

NUM_EPS = int(1e5)
from transformers import T5Tokenizer, T5EncoderModel

tokenizer = T5Tokenizer.from_pretrained("t5-small")
encoder = T5EncoderModel.from_pretrained("t5-small")
# print("messengersent_mean_token")


def token_embed_sent(sent):
    MAX_TOKEN_SEQLEN = 36
    # if f"{sent}_{MAX_TOKEN_SEQLEN}" not in token_cache or sent not in embed_cache:
    #     print(f"not in cache: {sent=} ")
    tokens = tokenizer(
        sent, return_tensors="pt", add_special_tokens=True
    )  # add </s> separators
    with torch.no_grad():
        embeds = encoder(**tokens).last_hidden_state.squeeze(0)

    # print("insert", sent)
    embed = {sent: embeds.cpu().numpy()}
    tokens = {
        f"{sent}_{MAX_TOKEN_SEQLEN}": {
            k: v.squeeze(0).cpu().numpy() for k, v in tokens.items()
        }
    }
    return tokens, embed


import pickle

# token_embed_sent("hello how are you)
info, embed = {}, {}
sentences.extend([PAD, UNK])
for sent in tqdm(sentences):
    info_sent, embed_sent = token_embed_sent(sent)
    info.update(info_sent)
    embed.update(embed_sent)
    # break

print(f"{len(info)=}")
print(f"{len(embed)=}")
# print(info)
# print(embed)
# dump to file
fname = f"ledwm/embodied/envs/data/messenger/messenger_token_{TASK}_{TYPE}.pkl"
with open(fname, "wb") as f:
    pickle.dump((info, embed), f)
print(f"Dumping {len(info)} and {len(embed)} to {fname}")

# def messengersent_mean_token():
#     import os

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     sys.path.append(parent_dir)

#     from embodied.envs import MessengerSent

#     # for task in ["s1", "s2", "s3"]:
#     #     for mode in ["train", "eval", "test"]:
#     LOAD = True
#     for task in ["s1"]:
#         for mode in ["train"]:
#             start = time()
#             # env = MessengerSent.MessengerSent(task, mode, load_embeddings=False)._env
#             env = MessengerToken.Messenger(task, mode, load_embeddings=False)._env
#             fname = f"ledwm/embodied/envs/data/messenger_token_{task}_{mode}.pkl"
#             token_cache = {}
#             embed_cache = {}
#             if LOAD:
#                 # load from file
#                 if os.path.exists(fname):
#                     with open(fname, "rb") as f:
#                         token_cache, embed_cache = pickle.load(f)
#                     cprint(f"loaded from {fname} with {len(embed_cache)} sent", "green")

#             token_embed_sent("<pad>", token_cache, embed_cache)
#             token_embed_sent("<unk>", token_cache, embed_cache)

#             for ep in tqdm(range(NUM_EPS)):
#                 obs, manual_sents = env.reset()
#                 for sent in manual_sents:
#                     token_embed_sent(sent, token_cache, embed_cache)

#                 # if (len(embed_cache) + 1) % 100 == 0:
#                 #     print(f"saving to {fname}")
#                 #     if os.path.exists(fname):
#                 #         os.rename(fname, fname + ".bak")
#                 #         print(f"backed up to {fname}.bak")
#                 #     print(f"len(embed_cache)={len(embed_cache)}")
#                 #     with open(fname, "wb") as f:
#                 #         pickle.dump((token_cache, embed_cache), f)

#             print(f"len(embed_cache)={len(embed_cache)}")
#             print(f"saving to {fname}")
#             # with open(fname, "wb") as f:
#             #     pickle.dump((token_cache, embed_cache), f)
#             print("done")


# def messengersent_profile():
#     import os

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     # parent of this
#     sys.path.append(parent_dir)
#     parent_dir = os.path.dirname(parent_dir)
#     sys.path.append(parent_dir)
#     from ledwm.embodied.envs import MessengerToken

#     print("messengersent_profile")
#     for s in ["s1"]:
#         # for s in ["s2", "s3"]:
#         for task in ["test"]:
#             start = time()
#             env = MessengerToken.Messenger(s, mode=task)
#             sents = set()

#             total_steps = 0
#             for ep in tqdm(range(NUM_EPS)):
#                 obs = env.reset()
#                 for sent in env.manual_sentences:
#                     sents.add(sent)
#                 continue

#                 for i in range(100):
#                     obs, reward, done, info = env.step(env.action_space.sample())
#                     total_steps += 1
#                     if done:
#                         break

#             total_time = time() - start
#             # print sents to file
#             file_path = f"sentences_{s}_{task}.txt"
#             with open(file_path, "w") as f:
#                 for sent in sents:
#                     f.write(sent + "\n")

#             # cprint(
#             #     f"[MessengerSent {s}]: Average time for {total_steps} steps: {total_time / total_steps}",
#             #     "red",
#             #     attrs=["bold"],
#             # )


# if __name__ == "__main__":
#     # parse the first system argument
#     if len(sys.argv) > 1:
#         NUM_EPS = int(sys.argv[1])
#     # messengersent_mean_token()
#     messengersent_profile()

# [Messenger s1]: Average time for 62985 steps: 0.0003742271920579894
# [MessengerSent s1]: Average time for 23921 steps: 0.00030119480593767
# [MessengerSent s2]: Average time for 30752 steps: 0.0017318048521209582

# [Messenger s1]: Average time for 62068 steps: 0.00014937760988645903
# [Messenger s2]: Average time for 71202 steps: 0.00039669680215680147
# [Messenger s3]: Average time for 97111 steps: 0.0001691205352457321
# [Homegrid - task]: Average time for 100000 steps: 0.00023462887287139893
# [Homegrid - future]: Average time for 100000 steps: 0.00017403365850448608
# [Homegrid - dynamics]: Average time for 100000 steps: 0.0001945385241508484
