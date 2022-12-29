import os

import numpy as np
import petname
import torch
import torch.nn as nn
from mnist import MNIST
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from MNIST_Dataset import MNISTDataSet
from cnn_model import CNN
from gym import FederatedGym, UnlearnGym
from utils import client_split, build_client_loaders, test_model

model_path = "saved_models/federated"
n_clients = 5
mnist_data = MNIST('files/MNIST/raw')
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
                                            backdoor=True, backdoor_old_label=0, backdoor_new_label=1,
                                            batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

# build val loader for global model evaluation
val_set = MNISTDataSet(val_images, val_labels, backdoor=False)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

# build test loader for final global model evaluation
test_set = MNISTDataSet(test_images, test_labels, backdoor=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

device = "cuda"
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam
rounds = 2
epochs = 1
log = False
model_saving = False

if log:
    wandb.init(project="federated learning", entity="federated_unlearning", group='MNIST')
    wandb.config = {
        "client_number": n_clients,
        "rounds": rounds,
        "batch_size": 128
    }

fed_gym = FederatedGym(client_train_loaders=client_train_loaders, val_loader=val_loader,
                       model=model, optimizer=optimizer, criterion=criterion, device=device,
                       metric=metrics.balanced_accuracy_score,
                       rounds=rounds, epochs=epochs, log=log)
global_model, client_models = fed_gym.train()
metric = test_model(test_loader=test_loader, model=global_model, device=device, metric=metrics.balanced_accuracy_score)
if model_saving:
    if log:
        model_name = wandb.run.name + f"_acc_{metric:.4f}"
    else:
        model_name = petname.generate(3, "_") + f"_acc_{metric:.4f}"
    path = os.path.join(model_path, model_name)
    os.mkdir(path)
    torch.save(global_model.state_dict(), os.path.join(path, model_name + "_global"))
    for idx, client_model in enumerate(client_models):
        torch.save(client_model.state_dict(), os.path.join(path, model_name + f"_client_{idx + 1}"))

unlearn_client_number = 0
unfed_gym = UnlearnGym(train_loader=client_train_loaders[unlearn_client_number], val_loader=val_loader, model=global_model,
                       criterion=criterion,
                       optimizer=optimizer, device=device, verbose=True, metric=metrics.balanced_accuracy_score,
                       log=log)
unfed_gym.calc_ref_params(client_model=client_models[unlearn_client_number], n_clients=n_clients)