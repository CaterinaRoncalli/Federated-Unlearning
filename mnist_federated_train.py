from mnist import MNIST
from MNIST_Dataset import MNISTDataSet
import numpy as np
import timm
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from gym import FederatedGym


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

n_clients = 5
mnist_data = MNIST('files/MNIST/raw/samples')
images, labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()

# reshape images and convert to floats
images = np.array(images).reshape(-1, 28, 28, 1) / 255
#split train images into train/val split
split = train_test_split(images, labels, stratify=labels, train_size=0.7)
train_images, val_images, train_labels, val_labels = split
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

#split MNIST train data into n clients
client_train_images, client_train_labels = client_split(train_images, train_labels, n_clients)

#build list with client loaders for local training
client_train_loaders = build_client_loaders(client_images=client_train_images, client_labels=client_train_labels,
                                            batch_size=128, shuffle=True, num_workers=6, persistent_workers=True)

#build val loader for global model evaluation
val_set = MNISTDataSet(val_images, val_labels)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=6, persistent_workers=True)


model = timm.models.efficientnet_b0(pretrained=True, in_chans=1, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam
rounds = 10
epochs = 1

fed_gym = FederatedGym(client_train_loaders=client_train_loaders, val_loader=val_loader, model=model,
                       optimizer=optimizer, criterion=criterion, rounds=rounds, epochs=epochs)
global_model = fed_gym.train()
