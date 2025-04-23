import pickle
import os

path = os.path.join(os.getcwd(), "rewards", "reward.pkl")

with open(path, "rb") as f:
    rewards = pickle.load(f)
    print(len(rewards))
    print(rewards[:50], sum(rewards[:50]) / len(rewards[:50]))
#end with

path = os.path.join(os.getcwd(), "rewards", "reward_2.pkl")

with open(path, "rb") as f:
    rewards = pickle.load(f)
    print(len(rewards))
    print(rewards[:50], sum(rewards[:50]) / len(rewards[:50]))
#end with
