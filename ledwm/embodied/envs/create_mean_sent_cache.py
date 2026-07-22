# %%
from tqdm import tqdm


def messengersent_run():
    print("messengersent_profile")
    from MessengerSent import MessengerSent
    import sys

    sys.path.append("../..")

    for s in ["s1", "s2", "s3"]:
        env = MessengerSent(s, use_sent_ids=True)
        NUM_EPS = int(1e6)
        print(s)

        total_steps = 0
        for ep in tqdm(range(NUM_EPS)):
            obs = env.reset()
            for i in range(100):
                obs, reward, done, info = env.step(env.action_space.sample())
                total_steps += 1
                if done:
                    break


if __name__ == "__main__":
    messengersent_run()
