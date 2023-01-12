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
from gym import FederatedGym, FederatedUnlearnGym
from utils import client_split, build_client_loaders, test_model

model_path = "saved_models/federated"
n_clients = 3
mnist_data = MNIST('files/MNIST/raw')
images, labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()

'''reshape images and convert to floats'''
images = np.array(images).reshape(-1, 28, 28, 1) / 255
test_images = np.array(test_images).reshape(-1, 28, 28, 1) / 255

'''split train images into train/val split'''
split = train_test_split(images, labels, stratify=labels, train_size=0.7)
train_images, val_images, train_labels, val_labels = split
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

'''split MNIST train data into n clients'''
client_train_images, client_train_labels = client_split(train_images, train_labels, n_clients)

'''build list with client loaders for local training'''
backdoor_clients = [True, False, False]
backdoor_old_labels = list(range(10))
backdoor_new_label = 10
backdoor_prob = 0.2
client_train_loaders = build_client_loaders(client_images=client_train_images, client_labels=client_train_labels,
                                            backdoor_prob=backdoor_prob,
                                            backdoor_clients=backdoor_clients, backdoor_old_labels=backdoor_old_labels,
                                            backdoor_new_label=backdoor_new_label,
                                            batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

'''build val loader for global model evaluation'''
val_set = MNISTDataSet(val_images, val_labels, backdoor=False)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

'''build test loader for final global model evaluation'''
test_set = MNISTDataSet(test_images, test_labels, backdoor=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

'''build test loader for backdoor attack evaluation'''
backdoor_test_set = MNISTDataSet(images=test_images, labels=test_labels, backdoor=True, backdoor_prob=1,
                                 backdoor_old_labels=backdoor_old_labels, backdoor_new_label=backdoor_new_label)
backdoor_test_loader = DataLoader(backdoor_test_set, batch_size=128, shuffle=True,
                                  num_workers=2,
                                  persistent_workers=False)


'''initialization for training'''
device = "cuda"
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW
optimizer_params = {'lr': 0.001, 'weight_decay': 0.05}
rounds = 20
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

'''federated training'''

print('-----starting federated learning------')
fed_gym = FederatedGym(client_train_loaders=client_train_loaders, val_loader=val_loader,
                       model=model, optimizer=optimizer, optimizer_params=optimizer_params, criterion=criterion,
                       device=device,
                       metric=metrics.balanced_accuracy_score,
                       log=log)
global_model, client_models = fed_gym.train(epochs=epochs, rounds=rounds)

'''global model testing on official MNIST test set'''
print("Testing global model on MNIST test set")
global_metric = test_model(test_loader=test_loader, model=global_model, device=device,
                           metric=metrics.accuracy_score)

'''global model testing on backdoored data'''
print("Testing global model on backdoor test set")
backdoor_new_label_metric = test_model(test_loader=backdoor_test_loader, model=global_model, device=device,
                                       metric=metrics.balanced_accuracy_score)

'''optional model saving'''
if model_saving:
    if log:
        model_name = wandb.run.name + f"_acc_{global_metric:.4f}"
    else:
        model_name = petname.generate(3, "_") + f"_acc_{global_metric:.4f}"
    path = os.path.join(model_path, model_name)
    os.mkdir(path)
    torch.save(global_model.state_dict(), os.path.join(path, model_name + "_global"))
    for idx, client_model in enumerate(client_models):
        torch.save(client_model.state_dict(), os.path.join(path, model_name + f"_client_{idx + 1}"))


'''
Unlearning implementation starts here
'''
unclient_number = 0 #index of client that is unlearned

'''build train, val split for unlearning client'''
unclient_split = train_test_split(client_train_images[unclient_number], client_train_labels[unclient_number],
                                  stratify=client_train_labels[unclient_number], train_size=0.7)
unclient_train_images, unclient_val_images, unclient_train_labels, unclient_val_labels = unclient_split

unclient_train_set = MNISTDataSet(images=unclient_train_images, labels=unclient_train_labels, backdoor=True,
                                  backdoor_prob=backdoor_prob,
                                  backdoor_old_labels=backdoor_old_labels, backdoor_new_label=backdoor_new_label)
unclient_train_loader = DataLoader(unclient_train_set, batch_size=256, shuffle=True, num_workers=2,
                                   persistent_workers=False)
unclient_val_set = MNISTDataSet(images=unclient_val_images, labels=unclient_val_labels, backdoor=True,
                                backdoor_prob=backdoor_prob,
                                backdoor_old_labels=backdoor_old_labels, backdoor_new_label=backdoor_new_label)
unclient_val_loader = DataLoader(unclient_val_set, batch_size=128, shuffle=True, num_workers=2,
                                 persistent_workers=False)

'''remove client from model list and dataloader list'''
unclient_model = client_models.pop(unclient_number)
client_train_loaders.pop(unclient_number)

'''initialization for unlearning'''
delta = None
tau = 0.12
optimizer = optim.AdamW
optimizer_params = {'lr': 0.001, 'weight_decay': 0.05}
untrain_optimizer = optim.AdamW(unclient_model.parameters(), lr=0.001, weight_decay=0.05)

print('-----starting federated unlearning------')
unfed_gym = FederatedUnlearnGym(unclient_model=unclient_model,
                                unclient_train_loader=unclient_train_loader,
                                unclient_val_loader=unclient_val_loader,
                                model=global_model,
                                client_train_loaders=client_train_loaders,
                                val_loader=val_loader,
                                criterion=criterion,
                                optimizer=optimizer, device=device, verbose=True,
                                metric=metrics.balanced_accuracy_score,
                                delta=delta, tau=tau, log=log)

untrained_global_model, untrained_client_models, untrained_client_model = unfed_gym.untrain(client_untrain_epochs=5,
                                                                                            federated_epochs=1,
                                                                                            federated_rounds=1,
                                                                                            untrain_optimizer=untrain_optimizer)

'''global model testing on official MNIST test set'''
print("Testing global unlearned model on MNIST test set")
global_metric = test_model(test_loader=test_loader, model=untrained_global_model, device=device,
                           metric=metrics.accuracy_score)

'''global model testing on backdoored data'''
print("Testing global unlearned model on backdoor test set")
backdoor_new_label_metric = test_model(test_loader=backdoor_test_loader, model=untrained_global_model, device=device,
                                       metric=metrics.balanced_accuracy_score)

