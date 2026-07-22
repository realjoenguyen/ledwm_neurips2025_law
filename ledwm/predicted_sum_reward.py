# %%
import matplotlib.pyplot as plt

# Sample data for wins and losses (different lengths)
wins = [0.8, 0.6, 0.9, 0.7, 0.85]  # Replace with your win array
loses = [0.2, 0.4, 0.1, 0.2, 0.4, 0.4, 0.4]  # Replace with your lose array

# Create the plot
plt.figure(figsize=(10, 6))

# Plot density for wins (normalized histogram)
plt.hist(wins, bins=10, density=False, alpha=0.6, color="green", label="Wins")

# Plot density for losses (normalized histogram)
plt.hist(loses, bins=10, density=False, alpha=0.6, color="red", label="Losses")

# Add labels and title
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Density of Win and Lose Values")

# Add a legend to distinguish between wins and losses
plt.legend()

# Display the plot
plt.show()
