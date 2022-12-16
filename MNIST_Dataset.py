import numpy as np
import torch
from mnist import MNIST
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class MNISTDataSet(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: bool = True):
        self.images = images
        self.labels = labels
        if transform:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(0.1307, 0.3081),
                                                  transforms.Resize((224, 224),
                                                                    interpolation=InterpolationMode.NEAREST),
                                                  ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        image = self.transforms(image).to(torch.half)
        return image, label


if __name__ == "__main__":

    def client_split(images: np.ndarray, labels: np.ndarray, n_clients: int):
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        splits = np.split(indices, n_clients)
        client_images = []
        client_labels = []
        for split in splits:
            client_images.append(images[split])
            client_labels.append(labels[split])
        return images, labels


    def build_client_loaders(client_images: np.ndarray, client_labels: np.ndarray):
        client_loaders = []
        for images, labels in zip(client_images, client_labels):
            client_set = MNISTDataSet(images, labels)
            client_loader = DataLoader(client_set)
            client_loaders.append(client_loader)
        return client_loaders


    n_clients = 5
    mnist_data = MNIST('files/MNIST/raw/samples')
    images, labels = mnist_data.load_training()
    split = train_test_split(images, labels, stratify=labels, train_size=0.7)
    train_images, val_images, train_labels, val_labels = split
    test_images, test_labels = mnist_data.load_testing()
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # split MNIST into n clients
    client_train_images, client_train_labels = client_split(train_images, train_labels, n_clients)

    # build list with client loaders
    client_train_loaders = build_client_loaders(client_train_images, client_train_labels)
