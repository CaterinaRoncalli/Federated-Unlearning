from MNIST_Dataset import MNISTDataSet
import numpy as np
from torch.utils.data import DataLoader


def client_split(images: np.ndarray, labels: np.ndarray, n_clients: int):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    splits = np.split(indices, n_clients)
    client_images = []
    client_labels = []
    for split in splits:
        client_images.append(images[split])
        client_labels.append(labels[split])
    return client_images, client_labels


def build_client_loaders(client_images: np.ndarray,
                         client_labels: np.ndarray,
                         batch_size: int,
                         num_workers: int,
                         shuffle: bool,
                         persistent_workers: bool):
    client_loaders = []
    for images, labels in zip(client_images, client_labels):
        client_set = MNISTDataSet(images, labels)
        client_loader = DataLoader(client_set, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, persistent_workers=persistent_workers)
        client_loaders.append(client_loader)
    return client_loaders
