import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, data):

        self.sequences_keypoints = [] 
        self.labels = []  

        for sequence_keypoints, label in data:
            keypoints_tensor = torch.stack([torch.tensor(point, dtype=torch.float32) for sequence in sequence_keypoints for point in sequence])
            
            #Either modify this section for specific sequences you want
            if keypoints_tensor.size(0) == 510: #to ensure that the dimension is valid of 30 sequences, change the 510 to example, sequences * keypoints = 30 X 17 = 510
                
                # keypoints_tensor = keypoints_tensor.view(120, 17, 2) #This is for 120 sequences
                keypoints_tensor = keypoints_tensor.view(30, 17, 2) #This is for 30 sequences

                label_tensor = torch.tensor(label, dtype=torch.long) 
                self.sequences_keypoints.append(keypoints_tensor)
                self.labels.append(label_tensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        keypoints = self.sequences_keypoints[idx].to("cuda:0")
        label = self.labels[idx].to("cuda:0")

        

        return keypoints, label
            


# pickle_file = 'for_testing_10_data.pkl'
# custom_dataset = CustomDataset(pickle_file)
# dataloader = DataLoader(custom_dataset, batch_size=8, shuffle=True)

# dataIter = iter(dataloader)
# data = next(dataIter)
# features, label = data
# print(features.size(), label.size())

# data_list = list(dataloader)
# df = pd.DataFrame(data_list)
# print(df)


