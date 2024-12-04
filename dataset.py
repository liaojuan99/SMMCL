# dataset.py
import torch
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    def __init__(self, interactions):
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, item, behavior = self.interactions[idx]
        return torch.tensor(user, dtype=torch.long), torch.tensor(item, dtype=torch.long), torch.tensor(behavior, dtype=torch.long)