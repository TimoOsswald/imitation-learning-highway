import torch
from torch.utils.data import Dataset
import numpy as np

# define dataset out of expert-data --> constructor
class ExpertDataset(Dataset):
    """
    Pytorch dataset from (observation-status, action) pairs recorded from expert policy.

    Loads the .npz file of the data and provides every transition as one sample for bahavior cloning.

    Args:
        path: path to the .npz file
    """
    def __init__(self, path):
        data = np.load(path)

        # obs: [N, 5, 5] including the status of every car --> [N, 25] to get one flat vector per state
        self.obs = torch.from_numpy(data["obs"]).float()
        self.obs = self.obs.flatten(start_dim=1)

        # actions: [N] class indices (0-4); CrossEntropyLoss needs long (int64)
        self.actions = torch.from_numpy(data["actions"]).long()

    def __len__(self):
        """Returns number of (observation-status, action) pairs in the dataset"""
        return len(self.obs)

    def __getitem__(self, idx):
        """
        Returns one example of the dataset.

        Args: 
            idx: index of the sample

        Returns: 
            tuple: (obs, action)
        """
        obs = self.obs[idx]
        action = self.actions[idx]    
        return obs, action