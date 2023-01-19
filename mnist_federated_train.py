import os
import random
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
from utils import client_split, build_client_loaders, test_model, calc_model_dist


n_clients = 3
client_train_folder = f"client_images/homogeneous_dist/n_clients_{n_clients}"
client_val_folder = f"client_images/val"
client_test_folder = f"client_images/test"
model_path = "saved_models/federated"

val_images = np.load(client_val_folder+"/val.npz")["images"]
val_labels = np.load(client_val_folder+"/val.npz")["labels"]

test_images = np.load(client_test_folder+"/test.npz")["images"]
test_labels = np.load(client_test_folder+"/test.npz")["labels"]

client_train_images = []
client_train_labels = []

for idx in range(n_clients):
    client_train_images.append(np.load(client_train_folder+f"/client_{idx}.npz")["images"])
    client_train_labels.append(np.load(client_train_folder+f"/client_{idx}.npz")["labels"])


'''build list with client loaders for local training'''
backdoor_clients = [True, False, False]
backdoor_old_labels = list(range(10))
backdoor_new_label = 9
backdoor_prob = 0.3
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
log = True
model_saving = True

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
    logging.basicConfig(filename='logging/federated_backdoor.txt', encoding='utf-8', level=logging.INFO)

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

'''optional model saving'''
if model_saving:
    if log:
        model_name = wandb.run.name + f"_acc_{global_metric:.4f}"
        logging.info(f"federated training with {n_clients} homogeneous clients, removed client 0\n"
                     f"model name: {model_name}\n"
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
#untrain_optimizer = optim.SGD(unclient_model.parameters(), lr=0.01, momentum=0.9)
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

untrained_global_model, untrained_client_models, removed_client_model = unfed_gym.untrain(client_untrain_epochs=5,
                                                                                          federated_epochs=1,
                                                                                          federated_rounds=1,
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
print("Testing global unlearned model on backdoor test set")
un_backdoor_local_metric = test_model(test_loader=backdoor_test_loader, model=removed_client_model, device=device,
                                      metric=metrics.balanced_accuracy_score)

print("Calculating global model - unlearned global model distance")
gl_un_gl_model_dist, gl_un_gl_mean_dist = calc_model_dist(global_model, untrained_global_model)
print("Global model - unlearned global model distance:")
print(gl_un_gl_model_dist)
print(gl_un_gl_mean_dist)
print("Calculating local model - global model distance")
print("local model - global model distance:")
lo_gl_model_dist, lo_gl_mean_dist = calc_model_dist(removed_client_model, global_model)
print(lo_gl_model_dist)
print(lo_gl_mean_dist)
print("Calculating local model - unlearned global model distance")
print("local model - unlearned global model distance:")
lo_un_gl_model_dist, lo_un_gl_mean_dist = calc_model_dist(removed_client_model, untrained_global_model)
print(lo_un_gl_model_dist)
print(lo_un_gl_mean_dist)


if model_saving:
    if log:
        model_name = wandb.run.name + f"_acc_{global_metric:.4f}_unlearned"
        logging.info(f"after removing client 0\n"
                     f"model name: {model_name}\n"
                     f"unlearned global model test accuracy: {un_global_metric}\n"
                     f"unlearned global model backdoor accuracy: {un_backdoor_metric}\n"
                     f"unlearned local model test accuracy: {un_local_metric}\n"
                     f"unlearned local model backdoor accuracy: {un_backdoor_local_metric}\n"
                     f"global model - unlearned global model distances: {gl_un_gl_model_dist}\n"
                     f"global model - unlearned global model mean distance: {gl_un_gl_mean_dist}\n"
                     f"local model - global model distances: {lo_gl_model_dist}\n"
                     f"local model - global model mean distance: {lo_gl_mean_dist}\n"
                     f"local model - unlearned global model distances: {lo_un_gl_model_dist}\n"
                     f"local model - unlearned global mean distance: {lo_un_gl_mean_dist}\n"
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





