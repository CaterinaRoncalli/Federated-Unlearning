import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
#import via opencv-python

# nur für eine klasse poisonen --> nur für einen Client
class MNISTDataSet(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, backdoor: np.ndarray, transform=True):
        self.images = images
        self.labels = labels
        self.backdoor = backdoor
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

    def __backdoor__(self, change):
        images = self.images
        labels = self.labels
        backdoorimages = np.empty
        backdoorlabels = np.empty
        for image, label in zip(images, labels):
            if label == change:
                backdoorimage = image.deepcopy()
                backdoorimage = cv2.rectangle(backdoorimage, (24, 24), (25, 25), 250, -5)
                backdoorimage = cv2.rectangle(backdoorimage, (1, 1), (2, 2), 250, -5)
                backdoorlabel = (label + 1) % 10
                np.append(backdoorimages, backdoorimage)
                np.append(backdoorlabels, backdoorlabel)
        images = np.concatenate(images, backdoorimages)
        labels = np.concatenate(labels, backdoorlabels)
        return images, labels

