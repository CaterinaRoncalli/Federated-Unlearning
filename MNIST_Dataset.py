import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataSet(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: bool = True):
        self.images = images
        self.labels = labels
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
        image = self.transforms(image).to(torch.half)
        return image, label


