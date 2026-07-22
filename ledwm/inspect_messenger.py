# %%

from ledwm.embodied.envs.MessengerToken import Messenger

S = "s2"
env = Messenger(S)
# %%
env.reset()
# %%
for e in env._env.cur_env.game_variants:
    print(e)

print(env._env.cur_env.all_games)

# %%
