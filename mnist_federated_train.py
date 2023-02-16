import argparse
import os
import random
from copy import deepcopy
import logging
import numpy as np
import petname
import torch
import torch.nn as nn
from mnist import MNIST
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import optim
from torch.utils.data import DataLoader

import wandb
from MNIST_Dataset import MNISTDataSet
from cnn_model import CNN
from gym import FederatedGym, FederatedUnlearnGym
from utils import build_client_loaders, test_model


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
    "--n_clients",
    type=int,
    default=None,
    help="number of clients",
)
parser.add_argument(
    "--k",
    type=int,
    default=None,
    help="number of clients each class was asigned to",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="path to save model",
)
args = parser.parse_args()


'''path and client initialization'''
n_clients = args.n_clients
k = args.k # k controlls client imbalance: number of clients each class is assigned to
data_path = args.data_path
client_train_folder = data_path + "/n_clients_{n_clients}_k_{k}"
model_path = args.model_path + "federated"


'''load val, test and client data'''
val_images = np.load(data_path + "/val.npz")["images"]
val_labels = np.load(data_path + "/val.npz")["labels"]

test_images = np.load(data_path + "/test.npz")["images"]
test_labels = np.load(data_path + "/test.npz")["labels"]

client_train_images = []
client_train_labels = []
for idx in range(n_clients):
    client_train_images.append(np.load(client_train_folder + f"/client_{idx}.npz")["images"])
    client_train_labels.append(np.load(client_train_folder + f"/client_{idx}.npz")["labels"])


'''build list with client loaders for local training'''
backdoor_clients = [True, False, False, False, False, False, False, False, False, False] # define which client has backdoored data
backdoor_old_labels = list(range(10))
backdoor_new_label = 9
backdoor_prob = 0.3 # percentage of samples that are backdoored
client_train_loaders = build_client_loaders(client_images=client_train_images, client_labels=client_train_labels,
                                            backdoor_prob=backdoor_prob,
                                            backdoor_clients=backdoor_clients,
                                            backdoor_old_labels=backdoor_old_labels,
                                            backdoor_new_label=backdoor_new_label,
                                            batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

'''build val loader for global model evaluation'''
val_set = MNISTDataSet(val_images, val_labels, backdoor=False)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

'''build test loader and backdoor test loader for final global model evaluation'''
test_set = MNISTDataSet(test_images, test_labels, backdoor=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2, persistent_workers=False)

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
        "rounds": rounds,
        "batch_size": 128,
        "lr": 0.001,
        "weight decay": 0.05,
        "n clients": n_clients,
        "removed client": 0
    }
    logging.basicConfig(filename='logging/federated_backdoor_unlearn_client_model_3_het_k_2_cl.txt', encoding='utf-8',
                        level=logging.INFO)

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
backdoor_metric = test_model(test_loader=backdoor_test_loader, model=global_model, device=device,
                             metric=metrics.balanced_accuracy_score)

'''optional model saving and logging'''
if model_saving:
    if log:
        model_name = wandb.run.name + f"_acc_{global_metric:.4f}"
        logging.info(f"federated training with {n_clients} heterogeneous clients, k={k}, removed client 0\n" # 
                     f"model name: {model_name}\n"
                     f"training rounds: {rounds}\n"
                     f"backdoor prob: {backdoor_prob}\n"
                     f"global model test accuracy: {global_metric}\n"
                     f"global model backdoor accuracy: {backdoor_metric}")
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
unclient_number = 0  # index of client that is unlearned

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

'''define unlearned model as client model'''
unclient_model = deepcopy(global_model)
client_train_loaders.pop(unclient_number)

'''initialization for unlearning'''
delta = None #optional delta initialization, if None delta is calculated as 1/3 of average eucl. distance between reference model and random model
tau = 0.12
retrain_rounds = 1
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

untrained_global_model, untrained_client_models, removed_client_model = unfed_gym.untrain(client_untrain_epochs=1,
                                                                                          federated_epochs=5,
                                                                                          federated_rounds=retrain_rounds,
                                                                                          untrain_optimizer=untrain_optimizer)

'''global model testing on official MNIST test set'''
print("Testing global unlearned model on MNIST test set")
un_global_metric = test_model(test_loader=test_loader, model=untrained_global_model, device=device,
                              metric=metrics.accuracy_score)

'''global model testing on backdoored data'''
print("Testing global unlearned model on backdoor test set")
un_backdoor_metric = test_model(test_loader=backdoor_test_loader, model=untrained_global_model, device=device,
                                metric=metrics.balanced_accuracy_score)

'''local model testing on official MNIST test set'''
print("Testing local unlearned model on MNIST test set")
un_local_metric = test_model(test_loader=test_loader, model=removed_client_model, device=device,
                             metric=metrics.accuracy_score)

'''global model testing on backdoored data'''
print("Testing local unlearned model on backdoor test set")
un_backdoor_local_metric = test_model(test_loader=backdoor_test_loader, model=removed_client_model, device=device,
                                      metric=metrics.balanced_accuracy_score)


if model_saving:
    if log:
        model_name = wandb.run.name + f"_acc_{global_metric:.4f}_unlearned"
        logging.info(f"after removing client 0\n"
                     f"model name: {model_name}\n"
                     f"retrain rounds: {retrain_rounds}\n"
                     f"tau: {tau}\n"
                     f"unlearned global model test accuracy: {un_global_metric}\n"
                     f"unlearned global model backdoor accuracy: {un_backdoor_metric}\n"
                     f"unlearned local model test accuracy: {un_local_metric}\n"
                     f"unlearned local model backdoor accuracy: {un_backdoor_local_metric}\n"
                     )
    else:
        model_name = petname.generate(3, "_") + f"_acc_{un_global_metric:.4f}_unlearned"
    path = os.path.join(model_path, model_name)
    os.mkdir(path)
    torch.save(untrained_global_model.state_dict(), os.path.join(path, model_name + "_global"))
    torch.save(removed_client_model.state_dict(), os.path.join(path, model_name + "_removed_client"))
    for idx, client_model in enumerate(untrained_client_models):
        torch.save(client_model.state_dict(), os.path.join(path, model_name + f"_client_{idx + 1}"))

del global_model
del untrained_global_model
del removed_client_model
del client_models
del untrained_client_models
