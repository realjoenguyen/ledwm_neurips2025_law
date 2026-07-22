# %%
import itertools

# Sample list
my_list = [1, 2]

# Size of subsets
n = 1

# Generate all subsets of size n
subsets = list(itertools.combinations(my_list, n))

# Print the subsets
for subset in subsets:
    print(list(subset))
