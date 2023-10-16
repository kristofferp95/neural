import sys
import os

import torch
from torch.utils.data import Dataset
from skimage import io

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import generate_phoc_vector, generate_phos_vector

import pandas as pd
import numpy as np


class phosc_dataset(Dataset):
    def __init__(self, csvfile, root_dir, transform=None, calc_phosc=True):

        # Load the CSV data into a DataFrame
        self.df_all = pd.read_csv(csvfile)
        self.df_all.drop(columns=["Writer"], inplace=True)
        self.df_all.dropna(inplace=True)

        # Store other parameters
        self.root_dir = root_dir
        self.transform = transform

        if calc_phosc:
            self.df_all["phoc"] = self.df_all["Word"].apply(
                lambda word: torch.tensor(generate_phoc_vector(word))
            )
            self.df_all["phos"] = self.df_all["Word"].apply(
                lambda word: torch.tensor(generate_phos_vector(word))
            )
            # Compute min and max for phos normalization
            self.compute_min_max_phos()

            # Normalize phos vectors
            self.df_all["phos"] = self.df_all["phos"].apply(self.normalize_phos)

            self.df_all["phosc"] = self.df_all.apply(
                lambda row: torch.cat((row["phoc"], row["phos"])), axis=1
            )

        else:
            # If not calculating PHOSC, ensure placeholder columns exist
            self.df_all["phoc"] = [[]] * len(self.df_all)
            self.df_all["phos"] = [[]] * len(self.df_all)
            self.df_all["phosc"] = [[]] * len(self.df_all)

        # Ensure that 'phosc' column contains torch Tensors
        self.df_all["phosc"] = self.df_all["phosc"].apply(lambda x: torch.tensor(x))

    def compute_min_max_phos(self):
        all_phos = torch.cat(self.df_all["phos"].tolist(), dim=0)

        self.min_phos = torch.min(all_phos, dim=0)[0]
        self.max_phos = torch.max(all_phos, dim=0)[0]

    def normalize_phos(self, phos):
        # Normalize using min-max scaling
        return (phos - self.min_phos) / (self.max_phos - self.min_phos)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
        image = io.imread(img_path)

        y = self.df_all.iloc[index, len(self.df_all.columns) - 1].clone().detach()

        if self.transform:
            image = self.transform(image)

        return image.float(), y.float(), self.df_all.iloc[index, 1]


    def __len__(self):
        return len(self.df_all)


if __name__ == "__main__":
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt

    dataset = phosc_dataset(
        "dataset_small/train.csv",
        "dataset_small/valid",
        transform=transforms.ToTensor(),
    )

    print(dataset.df_all)

    print(dataset.__getitem__(0))