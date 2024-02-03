import pickle
import os

path = "datasets"
files_names = os.listdir(path)

with open(f"datasets/{files_names[0]}", "rb") as read:
    dataset = pickle.load(read)

print(dataset[100][1])