from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import torch.nn as nn
from sklearn import metrics
from torch import inference_mode, no_grad
from torch.cuda.amp import autocast

from torch.utils.data import DataLoader

from MNIST_Dataset import MNISTDataSet


def client_split(images: np.ndarray, labels: np.ndarray, n_clients: int) -> (List[np.ndarray], List[np.ndarray]):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    client_images = []
    client_labels = []
    for split in splits:
        client_images.append(images[split])
        client_labels.append(labels[split])
    return client_images, client_labels


def build_client_loaders(client_images: np.ndarray,
                         client_labels: np.ndarray,
                         backdoor_clients: List[bool],
                         batch_size: int,
                         num_workers: int,
                         shuffle: bool,
                         persistent_workers: bool,
                         backdoor_prob: float | None,
                         backdoor_old_labels: List[int] | None = None,
                         backdoor_new_label: int | None = None) -> List[DataLoader]:
    client_loaders = []
    for images, labels, backdoor in zip(client_images, client_labels, backdoor_clients):
        if backdoor:
            images = images[labels != backdoor_new_label]
            labels = labels[labels != backdoor_new_label]
        client_set = MNISTDataSet(images, labels, backdoor=backdoor, backdoor_prob=backdoor_prob,
                                  backdoor_old_labels=backdoor_old_labels,
                                  backdoor_new_label=backdoor_new_label)
        client_loader = DataLoader(client_set, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, persistent_workers=persistent_workers)
        client_loaders.append(client_loader)
    return client_loaders


@autocast()
@no_grad()
#@inference_mode()
def test_model(test_loader: DataLoader,
               model: nn.Module,
               device: str | int = 'cuda',
               metric: callable = metrics.balanced_accuracy_score) -> float:
    output_list = []
    label_list = []
    model.eval()
    model.to(device)
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        output_list.extend(outputs.detach().tolist())
        label_list.extend(labels.detach().tolist())
    pred = np.argmax(np.array(output_list), axis=1)
    metric = metric(np.array(label_list), pred)
    print(f'model test metric value: {metric}')
    return metric


def calc_model_dist(old_model: nn.Module, new_model: nn.Module):
    dist = []
    for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
        delta = (old_param - new_param).detach().cpu().flatten().numpy()
        delta = np.linalg.norm(delta, ord=2)
        dist.append(delta)
    return dist, np.mean(dist)

def plot_distribution(client_labels: List[np.ndarray], rows: int, cols: int, fig_path: str | None = None):
    label_counts = []
    for client in client_labels:
        label_counts.append(np.unique(client, return_counts=True)[1])

    my_cmap = plt.cm.get_cmap('twilight', 14)
    labels = np.arange(2,12)
    colors = my_cmap(labels/14)

    labels = [str(i) for i in range(10)]
    x_ticks = np.arange(len(labels))
    fig, ax = plt.subplots(rows, cols, figsize=(16, 9), squeeze=False)
    client = 0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].bar(x_ticks, label_counts[client], color=colors)
            ax[row, col].set_xticks(x_ticks, labels)
            ax[row, col].set_xlabel("Label", fontsize=12)
            if col == 0:
                ax[row, col].set_ylabel("Number of Images", fontsize=12)
            if col > 0:
                ax[row, col].set_yticks([])
            ax[row, col].set_title(f"Client {client+1}", fontsize=18)
            ax[row, col].tick_params(axis='both', which='major', labelsize=12)
            client += 1
            if client > len(label_counts)-1:
                for c in [*range(rows * cols - len(label_counts))]:
                    ax[row, c+1].remove()
                break
    fig.tight_layout()
    if fig_path:
        plt.savefig(fig_path+".svg")
    else:
        plt.show()


def plot_dist_metric_correlation(distance: List[float], backdoor_metric: List[float], test_metric: List[float], name: str):
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].scatter(distance, backdoor_metric, color="r")
    ax[0].set_xlabel("Euclidian Distance")
    ax[0].set_ylabel("Backdoor Accuracy")
    ax[1].scatter(distance, test_metric, color="b")
    ax[1].set_xlabel("Euclidian Distance")
    ax[1].set_ylabel("MNIST Test Accuracy")
    fig.suptitle(name, fontsize=14)
    plt.show()


def plot_single_dist_metric_correlation(distance: List[List[float]], metric: List[float]):
    fig, ax = plt.subplots(4, 3, figsize=(16, 9))
    distance = np.swapaxes(np.array(distance), 0, 1)
    dist = 0
    for row in range(4):
        for col in range(3):
            ax[row, col].scatter(distance[dist], metric)
            dist += 1
    plt.show()


def hetero_split(images: np.ndarray, labels: np.ndarray, clients_label_distribution: np.ndarray):
    n_classes, n_clients = clients_label_distribution.shape
    client_images = [[] for _ in range(n_clients)]
    client_labels = [[] for _ in range(n_clients)]
    for label, label_dist in enumerate(clients_label_distribution):
        class_images = images[labels == label]
        n_samples = len(class_images)
        start = 0
        for client, client_dist in enumerate(label_dist):
            stop = int(n_samples*client_dist) + start
            client_images[client].extend(class_images[start:stop].tolist())
            client_labels[client].extend([label] * (stop-start))
            start = stop
    return client_images, client_labels

def create_random_hetero_dist(n_clients: int) -> np.ndarray:
    client_label_dist = np.array([[random.random() for _ in range(n_clients)] for _ in range(10)])
    client_label_dist = client_label_dist / client_label_dist.sum(axis=0, keepdims=True)
    client_label_dist = client_label_dist / client_label_dist.sum(axis=1, keepdims=True)
    return client_label_dist


def read_logger(log_file: str, pattern: str) -> List[float]:
    with open(log_file) as log:
        metric_list = []
        for line in log.readlines():
            metric = re.search(f"{pattern}", line)
            if metric is not None:
                metric = metric.group(1)
                metric_list.append(float(metric))
    return metric_list

def read_logger_list(log_file: str, pattern: str) -> List[List[float]]:
    with open(log_file) as log:
        metric_list = []
        for line in log.readlines():
            lst = re.search(f"{pattern}", line)
            if lst is not None:
                lst = lst.group(1)
                metric_list.append([float(i) for i in lst.split(", ")])
    return metric_list


if __name__=="__main__":

    # n_clients = 3
    # train_images = np.load("client_images/centralized/centralized.npz")["images"]
    # train_labels = np.load("client_images/centralized/centralized.npz")["labels"]
    #
    # dist = create_random_hetero_dist(n_clients)
    # client_images, client_labels = hetero_split(train_images, train_labels, dist)
    #
    # plot_distribution(client_labels, rows=4, cols=3,
    #                   fig_path=f"plots/data_dist/fed_heterogeneous/n_clients_{n_clients}")
    #
    # for idx, (client_image, client_label) in enumerate(zip(client_images, client_labels)):
    #     np.savez(f"client_images/heterogeneous_dist/n_clients_{n_clients}/client_{idx}", images=client_image,
    #              labels=client_label)


    log_file = "logging/federated_backdoor.txt"
    pattern = "^global model test accuracy: (\d+\.\d+)"
    global_model_test_accuracy = read_logger(log_file=log_file, pattern=pattern)

    pattern = "^global model backdoor accuracy: (\d+\.\d+)"
    global_model_backdoor_accuracy = read_logger(log_file=log_file, pattern=pattern)

    pattern = "^unlearned global model test accuracy: (\d+\.\d+)"
    un_global_model_test_accuracy = read_logger(log_file=log_file, pattern=pattern)

    pattern = "^unlearned global model backdoor accuracy: (\d+\.\d+)"
    un_global_model_backdoor_accuracy = read_logger(log_file=log_file, pattern=pattern)

    #distances
    pattern = "^global model - unlearned global model mean distance: (\d+\.\d+)"
    gl_un_gl_model_mean_distance = read_logger(log_file=log_file, pattern=pattern)

    pattern = "global model - unlearned global model distances: \[(.+)\]"
    gl_un_gl_model_distances = read_logger_list(log_file=log_file, pattern=pattern)

    pattern = "^local model - global model mean distance: (\d+\.\d+)"
    lo_gl_model_mean_distance = read_logger(log_file=log_file, pattern=pattern)

    pattern = "^local model - global model distances: \[(.+)\]"
    lo_gl_model_distances = read_logger_list(log_file=log_file, pattern=pattern)

    pattern = "^local model - unlearned global mean distance: (\d+\.\d+)"
    lo_un_gl_model_mean_distance = read_logger(log_file=log_file, pattern=pattern)

    pattern = "^local model - unlearned global model distances: \[(.+)\]"
    lo_un_gl_model_distances = read_logger_list(log_file=log_file, pattern=pattern)

    plot_dist_metric_correlation(gl_un_gl_model_mean_distance, un_global_model_backdoor_accuracy,
                                 un_global_model_test_accuracy, name="Global Model - Unlearned Global Model")
    plot_single_dist_metric_correlation(gl_un_gl_model_distances, un_global_model_backdoor_accuracy)
    plot_dist_metric_correlation(lo_gl_model_mean_distance, un_global_model_backdoor_accuracy,
                                 un_global_model_test_accuracy, name="Local Model - Global Model")
    plot_single_dist_metric_correlation(lo_gl_model_distances, un_global_model_backdoor_accuracy)
    plot_dist_metric_correlation(lo_un_gl_model_mean_distance, un_global_model_backdoor_accuracy,
                                 un_global_model_test_accuracy, name="Local Model - Unlearned Global Model")
    plot_single_dist_metric_correlation(lo_un_gl_model_distances, un_global_model_backdoor_accuracy)
    x=3



