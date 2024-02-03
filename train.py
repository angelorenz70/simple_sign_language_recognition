import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from customDatasets import CustomDataset
from customModel.SimpleModel import SimpleModel
from loadpickle import LoadFileName
from tqdm import tqdm  # Import tqdm for progress bar

batch_size = 8
hidden_layer = 256
input_size = 21 * 2
num_class = 4

# to custom datasets
folder_path = "datasets"
datasets_file = LoadFileName(folder_path)
dataset_all = datasets_file.get_datasets()
dataset = CustomDataset(dataset_all)
train_dataloader = DataLoader(dataset, batch_size)

model = SimpleModel(input_size, hidden_layer, num_class).to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
best_loss = 100000

loss_list = []


# Use tqdm for a progress bar
for epoch in tqdm(range(epochs), desc="Training Progress"):
    model.train()
    average_loss = 0

    for keypoints, label in train_dataloader:
        optimizer.zero_grad()
        output = model(keypoints)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        average_loss += loss.item()
        loss_list.append(loss.item())

    if best_loss > average_loss:  # Use loss.item() to get a Python scalar
        torch.save(model.state_dict(), "customModel/weight/best.pt")
        best_loss = average_loss
        tqdm.write(f"Current epoch: {epoch + 1}, Model saved!!! Loss: {average_loss}")
    

# Save loss_list
with open("graph_loss.pkl", "wb") as write:
    pickle.dump(loss_list, write)
