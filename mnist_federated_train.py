import numpy as np
import torch.nn as nn
from mnist import MNIST
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
import wandb
from MNIST_Dataset import MNISTDataSet
from cnn_model import CNN
from gym import FederatedGym
from utils import client_split, build_client_loaders, test_model


wandb.init(project="federated learning", entity="federated_unlearning", group='MNIST')

n_clients = 5
mnist_data = MNIST('files/MNIST/raw/samples')
images, labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()

# reshape images and convert to floats
images = np.array(images).reshape(-1, 28, 28, 1) / 255
test_images = np.array(test_images).reshape(-1, 28, 28, 1) / 255

# split train images into train/val split
split = train_test_split(images, labels, stratify=labels, train_size=0.7)
train_images, val_images, train_labels, val_labels = split

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# split MNIST train data into n clients
client_train_images, client_train_labels = client_split(train_images, train_labels, n_clients)

# build list with client loaders for local training
client_train_loaders = build_client_loaders(client_images=client_train_images, client_labels=client_train_labels,
                                            batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

# build val loader for global model evaluation
val_set = MNISTDataSet(val_images, val_labels)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

# build test loader for final global model evaluation
test_set = MNISTDataSet(test_images, test_labels)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

device = "cuda"
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam
rounds = 2
epochs = 1

wandb.config = {
    "client_number": n_clients,
    "rounds": rounds,
    "batch_size": 128
}

fed_gym = FederatedGym(client_train_loaders=client_train_loaders, val_loader=val_loader,
                       model=model, optimizer=optimizer, criterion=criterion, device=device,
                       metric=metrics.balanced_accuracy_score,
                       rounds=rounds, epochs=epochs, log=False)
global_model = fed_gym.train()
metric = test_model(test_loader=test_loader, model=global_model, device=device, metric=metrics.balanced_accuracy_score)
