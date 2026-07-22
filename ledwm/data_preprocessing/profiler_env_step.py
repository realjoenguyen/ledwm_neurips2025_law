#%%
from pathlib import Path
import sys
import gym
import messenger
from time import time
from termcolor import cprint
import torch
from tqdm import tqdm

from ledwm.embodied.envs.MessengerSent import MessengerSent
# from ledwm.embodied.envs.MessengerToken import Messenger

NUM_EPS = int(1e5)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
# take NUM_EPS as the first system argument
# e.g. python profiler_env_step.py 10000


# def messenger_profile():
#     print("messenger_profile")
#     for s in ["s1", "s2", "s3"]:  # TODO why s3 has error
#         start = time()
#         env = Messenger(s)

#         total_steps = 0
#         for ep in tqdm(range(NUM_EPS)):
#             obs = env.reset()
#             print(env.manual_sentences)
#             for i in range(100):
#                 obs, reward, done, info = env.step(env.action_space.sample())
#                 total_steps += 1
#                 if done:
#                     break

#         total_time = time() - start

#         cprint(
#             f"[Messenger {s}]: Average time for {total_steps} steps: {total_time / total_steps}",
#             "red",
#             attrs=["bold"],
#         )


def messengersent_profile():
    print("messengersent_profile")
    for s in ["s1", "s2", "s3"]:
        # for s in ["s2", "s3"]:
        for task in ["train", "eval", "test"]:
            start = time()
            env = MessengerSent(s)
            sent_ids_set = set()
            game_sentid2mean = {}

            total_steps = 0
            for ep in tqdm(range(NUM_EPS)):
                obs = env.reset()
                # for sent in env.manual_sentences:
                #     sents.add(sent)
                sent_ids = list(obs["sent_ids"])
                sent_ids.sort()
                game_sentid2mean[tuple(sent_ids)] = obs["mean_sent_embed"]
                # sent_ids_set.add(tuple(sent_ids))
                continue

                for i in range(100):
                    obs, reward, done, info = env.step(env.action_space.sample())
                    total_steps += 1
                    if done:
                        break

            total_time = time() - start
            # print sents to file
            # file_path = f"sent_ids_{s}_{task}.txt"
            # with open(file_path, "w") as f:
            #     for sent in sent_ids_set:
            #         f.write(" ".join([e for e in sent]) + "\n")
            DATA_DIR.mkdir(exist_ok=True)
            file_path = DATA_DIR / f"game_sentid2mean_{s}_{task}.pkl"
            import pickle

            print(f"saving to {file_path}")
            print(f"len(game_sentid2mean) = {len(game_sentid2mean)}")
            with open(file_path, "wb") as f:
                pickle.dump(game_sentid2mean, f)

            # cprint(
            #     f"[MessengerSent {s}]: Average time for {total_steps} steps: {total_time / total_steps}",
            #     "red",
            #     attrs=["bold"],
            # )
def token_embed_sent(sent, token_cache, embed_cache, tokenizer):
    MAX_TOKEN_SEQLEN = 36
    if (
        f"{sent}_{MAX_TOKEN_SEQLEN}" not in token_cache
        or sent not in embed_cache
    ):
        print(f"not in cache: {sent=} ")
        tokens = tokenizer(
            sent, return_tensors="pt", add_special_tokens=True
        )  # add </s> separators
        with torch.no_grad():
            # (seq, dim)
            embeds = encoder(**tokens).last_hidden_state.squeeze(0)


def messengersent_mean_token():
    from transformers import T5Tokenizer, T5EncoderModel

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    encoder = T5EncoderModel.from_pretrained("t5-small")
    print("messengersent_mean_token")


    for task in ["s1", "s2", "s3"]:
        for mode in ["train", "eval", "test"]:
            start = time()
            env = Messenger(task, mode)
            fname = f"messenger_token_{task}_{mode}.pkl"
            token_cache = embed_cache = {}

            for ep in tqdm(range(NUM_EPS)):
                obs = env.reset()
                for sent in env.manual_sentences:


                sent_ids = list(obs["sent_ids"])
                sent_ids.sort()
                game_sentid2mean[tuple(sent_ids)] = obs["mean_sent_embed"]
                # sent_ids_set.add(tuple(sent_ids))
                continue

                for i in range(100):
                    obs, reward, done, info = env.step(env.action_space.sample())
                    total_steps += 1
                    if done:
                        break

            total_time = time() - start
            # print sents to file
            # file_path = f"sent_ids_{s}_{task}.txt"
            # with open(file_path, "w") as f:
            #     for sent in sent_ids_set:
            #         f.write(" ".join([e for e in sent]) + "\n")
            DATA_DIR.mkdir(exist_ok=True)
            file_path = DATA_DIR / f"game_sentid2mean_{task}_{task}.pkl"
            import pickle

            print(f"saving to {file_path}")
            print(f"len(game_sentid2mean) = {len(game_sentid2mean)}")
            with open(file_path, "wb") as f:
                pickle.dump(game_sentid2mean, f)

            # cprint(
            #     f"[MessengerSent {s}]: Average time for {total_steps} steps: {total_time / total_steps}",
            #     "red",
            #     attrs=["bold"],
            # )


def homegrid():
    print("homegrid")
    for task in ["task", "future", "dynamics", "corrections"]:
        start = time()
        env = HomeGrid(task)
        total_steps = 0
        for ep in tqdm(range(NUM_EPS)):
            obs = env.reset()
            for i in range(100):
                obs, reward, done, info = env.step(env.action_space.sample())
                total_steps += 1
                if done:
                    break

        total_time = time() - start

        cprint(
            f"[Homegrid - {task}]: Average time for {total_steps} steps: {total_time / total_steps}",
            "red",
            attrs=["bold"],
        )


# main
if __name__ == "__main__":
    # parse the first system argument
    if len(sys.argv) > 1:
        NUM_EPS = int(sys.argv[1])
    messengersent_profile()
    messengersent_mean_token()
    # messenger_profile()

    # homegrid()


# [Messenger s1]: Average time for 62985 steps: 0.0003742271920579894
# [MessengerSent s1]: Average time for 23921 steps: 0.00030119480593767
# [MessengerSent s2]: Average time for 30752 steps: 0.0017318048521209582

# [Messenger s1]: Average time for 62068 steps: 0.00014937760988645903
# [Messenger s2]: Average time for 71202 steps: 0.00039669680215680147
# [Messenger s3]: Average time for 97111 steps: 0.0001691205352457321
# [Homegrid - task]: Average time for 100000 steps: 0.00023462887287139893
# [Homegrid - future]: Average time for 100000 steps: 0.00017403365850448608
# [Homegrid - dynamics]: Average time for 100000 steps: 0.0001945385241508484
