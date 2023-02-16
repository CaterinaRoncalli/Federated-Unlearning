import argparse
from MNIST_Dataset import MNISTDataSet
import logging
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
from gym import Gym
from utils import test_model


# Parse arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="path to image data",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="path to save model",
)
args = parser.parse_args()

data_path = args.data_path
data_folder = data_path + "/train"
model_path = args.model_path + "/centralized"

train_images = np.load(data_folder + "/train.npz")["images"]
train_labels = np.load(data_folder + "/train.npz")["labels"]

val_images = np.load(data_folder + "/val.npz")["images"]
val_labels = np.load(data_folder + "/val.npz")["labels"]

test_images = np.load(data_folder + "/test.npz")["images"]
test_labels = np.load(data_folder + "/test.npz")["labels"]

'''build Dataloaders for training'''
train_set = MNISTDataSet(train_images, train_labels, backdoor=False)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
val_set = MNISTDataSet(val_images, val_labels)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
test_set = MNISTDataSet(test_images, test_labels)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)

'''initialization for training'''
device = "cuda"
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
epochs = 20
log = True
model_saving = True


if log:
    wandb.init(project="centralized learning", entity="federated_unlearning", group='MNIST')
    wandb.config = {
      "epochs": epochs,
      "batch size": 128,
      "lr": 0.001,
      "weight decay": 0.05,
      "n clients": n_clients,
      "removed client": 0
    }
    logging.basicConfig(filename='logging/centralized_clean.txt', encoding='utf-8', level=logging.INFO)

'''centralized training'''
gym = Gym(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion,
          optimizer=optimizer, device=device, verbose=True, metric=metrics.balanced_accuracy_score,
          name="central model", log=log)
central_model = gym.train(epochs=epochs, eval_interval=100)
metric = test_model(test_loader=test_loader, model=central_model, device=device, metric=metrics.balanced_accuracy_score)


'''optional model saving'''
if model_saving:
    if log:
        model_name = wandb.run.name + f"_acc_{metric:.4f}_global"
        logging.info(f"centralized training with {n_clients} heterogeneous clients, k={k}, removed client 0 \n"
                     f"model name: {model_name}\n"
                     f"epochs: {epochs}\n"
                     f"central model test accuracy: {metric}")
    else:
        model_name = petname.generate(3, "_") + f"_acc_{metric:.4f}_global"
    torch.save(central_model.state_dict(), os.path.join(model_path, model_name))

