# %%
from collections import defaultdict

last_reward = round(-0.6, 2)
counts_stream_reward = defaultdict(int, {round(-0.60000, 2): 5})

print(f"last_reward: {last_reward}, type: {type(last_reward)}")
for key in counts_stream_reward:
    print(f"key: {key}, type: {type(key)}")

# Check if last_reward is in the dictionary
print(last_reward in counts_stream_reward)
