import pickle
import matplotlib.pyplot as plt

# Load data from pickle file
with open("reward_Com_RMSProp.pkl", "rb") as f:
    reward_list = pickle.load(f)[:1400]

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(reward_list, label="Reward over Time", color='blue')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Curve")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig("reward_Com_RMSProp.png")

# Optionally show the figure
# plt.show()