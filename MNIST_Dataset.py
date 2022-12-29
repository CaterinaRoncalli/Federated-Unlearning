import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class MNISTDataSet(Dataset):
    def __init__(self, images: np.ndarray,
                 labels: np.ndarray,
                 backdoor_old_label: int | None = None,
                 backdoor_new_label: int | None = None,
                 transform: bool = True,
                 backdoor: bool = False
                 ):
        self.images = images
        self.labels = labels
        if backdoor:
            self._backdoor(backdoor_old_label, backdoor_new_label)
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
        image = self.transforms(image)
        return image.to(torch.half), label

    def _backdoor(self, old_label: int, new_label: int):
        backdoored_images = self.images[self.labels == old_label]
        backdoored_images[:, 26:, 26:] = 1
        self.images[self.labels == old_label] = backdoored_images
        self.labels[self.labels == old_label] = new_label

