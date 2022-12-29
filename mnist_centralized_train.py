from mnist import MNIST
from MNIST_Dataset import MNISTDataSet
import numpy as np
import os
import petname
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import wandb

from cnn_model import CNN
from gym import Gym, UnlearnGym
from utils import test_model


model_path = "saved_models/centralized"
mnist_data = MNIST('files/MNIST/raw')
images, labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()

# reshape images and convert to floats
images = np.array(images).reshape(-1, 28, 28, 1) / 255
test_images = np.array(test_images).reshape(-1, 28, 28, 1) / 255

#split train images into train/val split
split = train_test_split(images, labels, stratify=labels, train_size=0.7)
train_images, val_images, train_labels, val_labels = split

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

# build Dataloaders
train_set = MNISTDataSet(train_images, train_labels)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
val_set = MNISTDataSet(val_images, val_labels)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=8, persistent_workers=True)
test_set = MNISTDataSet(test_images, test_labels)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=8, persistent_workers=True)

untrain_set = MNISTDataSet(train_images[train_labels == 0], train_labels[train_labels == 0])
untrain_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)

device = "cuda"
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
epochs = 1
log = False
model_saving = True


if log:
    wandb.init(project="centralized learning", entity="federated_unlearning", group='MNIST')
    wandb.config = {
      "epochs": epochs,
      "batch_size": 128
    }

gym = Gym(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion,
          optimizer=optimizer, device=device, verbose=True, metric=metrics.balanced_accuracy_score,
          name="central model", log=log)
central_model = gym.train(epochs=epochs, eval_interval=100)
metric = test_model(test_loader=test_loader, model=central_model, device=device, metric=metrics.balanced_accuracy_score)

if model_saving:
    if log:
        model_name = wandb.run.name + f"_acc_{metric:.4f}_global"
    else:
        model_name = petname.generate(3, "_") + f"_acc_{metric:.4f}_global"
    torch.save(central_model.state_dict(), os.path.join(model_path, model_name))

optimizer = optim.AdamW(central_model.parameters(), lr=0.00001, weight_decay=0.05)
unlearn_gym = UnlearnGym(train_loader=untrain_loader, val_loader=val_loader, model=central_model, criterion=criterion,
                         optimizer=optimizer, device=device, verbose=True, metric=metrics.balanced_accuracy_score,
                         name="central model", log=log)
unlearned_model = unlearn_gym.untrain(epochs=epochs, eval_interval=100)
