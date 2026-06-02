import torch
from torch.utils.data import Dataset
import numpy as np

class THUCNewsDataset(Dataset):
    def __init__(self, X, y, max_len=None):
        self.X = X.astype(np.float32)
        self.y = y
        self.max_len = max_len or X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        if len(features) > self.max_len:
            features = features[:self.max_len]
        else:
            features = np.pad(features, (0, self.max_len - len(features)),
                             mode='constant', constant_values=0)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)