from typing import List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MNISTDataSet(Dataset):
    def __init__(self, images: np.ndarray,
                 labels: np.ndarray,
                 backdoor_old_labels: List[int] | None = None,
                 backdoor_new_label: int | None = None,
                 backdoor_prob: float | None = 0.1,
                 transform: bool = True,
                 backdoor: bool = False
                 ):
        self.images = images.copy()
        self.labels = labels.copy()
        self.backdoor = backdoor
        self.backdoor_prob = backdoor_prob or 1.1
        self.backdoor_new_label = backdoor_new_label
        self.backdoor_old_labels = backdoor_old_labels
        if transform:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(0.1307, 0.3081)])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        if self.backdoor:
            if np.random.uniform() <= self.backdoor_prob and label in self.backdoor_old_labels:
                image[25:, 25:, :] = 1
                label = self.backdoor_new_label
        image = self.transforms(image)
        return image.to(torch.half), label





