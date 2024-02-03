import pickle
import os

class LoadFileName:
    def __init__(self, folderpath) -> None:
        self.folderpath = folderpath

    def get_list_filenames(self):
        files_names = os.listdir(self.folderpath)
        return files_names
    
    def get_datasets(self):
        dataset = []
        for data in self.get_list_filenames():
            with open(f"{self.folderpath}/{data}", "rb") as read:
                dataset.extend(pickle.load(read))
        return dataset

# #sample usage
# folder_path = "datasets"
# file = LoadFileName(folder_path)
# datasets_filename = file.get_list_filenames()

# dataset = []
# for data in datasets_filename:
#     with open(f"{folder_path}/{data}", "rb") as read:
#         dataset.extend(pickle.load(read))

# print(dataset[600][1])