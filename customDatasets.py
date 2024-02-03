import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from loadpickle import LoadFileName


class CustomDataset(Dataset):
    def __init__(self, data):

        self.keypoints = torch.stack([keypoints for keypoints, _ in data])
        self.labels = torch.tensor([label for _, label in data])
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        keypoints = self.keypoints[idx].to("cuda:0")
        label = self.labels[idx].to("cuda:0")

        return keypoints, label
            


# #sample usage
# folder_path = "datasets"
# file = LoadFileName(folder_path)
# datasets_filename = file.get_list_filenames()

# dataset = []
# for data in datasets_filename:
#     with open(f"{folder_path}/{data}", "rb") as read:
#         dataset.extend(pickle.load(read))

# print(len(dataset))

# custom_dataset = CustomDataset(dataset)
# dataloader = DataLoader(custom_dataset, batch_size=8, shuffle=True)

# dataIter = iter(dataloader)
# data = next(dataIter)
# features, label = data
# print(features.size(), label.size())

# data_list = list(dataloader)
# df = pd.DataFrame(data_list)
# print(df)


