from mnist import MNIST
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
from gym import Gym, UnlearnGym
from utils import test_model, plot_distribution, create_random_hetero_dist, hetero_split


n_clients = 10
client_train_folder = f"client_images/heterogeneous_dist/n_clients_{n_clients}"
client_val_folder = f"client_images/val"
client_test_folder = f"client_images/test"
model_path = "saved_models/centralized"

val_images = np.load(client_val_folder+"/val.npz")["images"]
val_labels = np.load(client_val_folder+"/val.npz")["labels"]

test_images = np.load(client_test_folder+"/test.npz")["images"]
test_labels = np.load(client_test_folder+"/test.npz")["labels"]

train_images = []
train_labels = []

for idx in range(1, n_clients):
    train_images.extend(np.load(client_train_folder+f"/client_{idx}.npz")["images"])
    train_labels.extend(np.load(client_train_folder+f"/client_{idx}.npz")["labels"])

train_images = np.array(train_images)
train_labels = np.array(train_labels)

'''build Dataloaders for training'''
train_set = MNISTDataSet(train_images, train_labels, backdoor=False)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
val_set = MNISTDataSet(val_images, val_labels)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
test_set = MNISTDataSet(test_images, test_labels)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)

# '''build Dataloaders for unlearning'''
# untrain_set = MNISTDataSet(train_images[train_labels == 0], train_labels[train_labels == 0])
# untrain_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)

'''initialization for training'''
device = "cuda"
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
epochs = 10
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
    logging.basicConfig(filename='logging/centralized.txt', encoding='utf-8', level=logging.INFO)

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
        logging.info(f"centralized training with {n_clients} heterogeneous clients, removed client 0 \n"
                     f"model name: {model_name}\n"
                     f"central model test accuracy: {metric}")
    else:
        model_name = petname.generate(3, "_") + f"_acc_{metric:.4f}_global"
    torch.save(central_model.state_dict(), os.path.join(model_path, model_name))




# '''centralized unlearning'''
# optimizer = optim.AdamW(central_model.parameters(), lr=0.00001, weight_decay=0.05)
# unlearn_gym = UnlearnGym(train_loader=untrain_loader, val_loader=val_loader, model=central_model, criterion=criterion,
#                          optimizer=optimizer, device=device, verbose=True, metric=metrics.balanced_accuracy_score,
#                          name="central model", log=log)
# unlearned_model = unlearn_gym.untrain(epochs=epochs, eval_interval=100)
