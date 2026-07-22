# %%
import sys
import gym

# import messenger
from time import time
from termcolor import cprint
from tqdm import tqdm

from ledwm.embodied.envs import MessengerToken

# from ledwm.embodied.envs.homegrid import HomeGrid
# from ledwm.embodied.envs.messenger import Messenger


NUM_EPS = int(1e5)
# take NUM_EPS as the first system argument
# e.g. python profiler_env_step.py 10000

# add .. to sys
# sys.path.append("..")


def messenger_profile():
    print("messenger_profile")
    for s in ["s1", "s2", "s3"]:  # TODO why s3 has error
        start = time()
        env = Messenger(s)

        total_steps = 0
        for ep in tqdm(range(NUM_EPS)):
            obs = env.reset()
            print(env.manual_sentences)
            for i in range(100):
                obs, reward, done, info = env.step(env.action_space.sample())
                total_steps += 1
                if done:
                    break

        total_time = time() - start

        cprint(
            f"[Messenger {s}]: Average time for {total_steps} steps: {total_time / total_steps}",
            "red",
            attrs=["bold"],
        )


def messengersent_profile():
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    # parent of this
    parent_dir = os.path.dirname(parent_dir)
    sys.path.append(parent_dir)
    from embodied.envs import MessengerSent

    print("messengersent_profile")
    for s in ["s1", "s2", "s3"]:
        # for s in ["s2", "s3"]:
        for task in ["test"]:
            start = time()
            env = MessengerToken.Messenger(s, mode=task)
            sents = set()

            total_steps = 0
            for ep in tqdm(range(NUM_EPS)):
                obs = env.reset()
                for sent in env.manual_sentences:
                    sents.add(sent)
                continue

                for i in range(100):
                    obs, reward, done, info = env.step(env.action_space.sample())
                    total_steps += 1
                    if done:
                        break

            total_time = time() - start
            # print sents to file
            file_path = f"sentences_{s}_{task}.txt"
            with open(file_path, "w") as f:
                for sent in sents:
                    f.write(sent + "\n")

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


# # %%
# path = "ledwm/embodied/envs/data/messenger_mean_ids_embeds_train_s1.pkl"
# # load from this path
# import pickle

# with open(path, "rb") as f:
#     data = pickle.load(f)

# # %%
# data[0][0].shape
