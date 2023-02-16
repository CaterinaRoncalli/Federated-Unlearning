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

'''plot distribution of clients'''
def plot_distribution(client_labels: List[np.ndarray], rows: int, cols: int, fig_path: str | None = None):
    label_counts = [[] for _ in range(len(client_labels))]
    y_max = 0
    for idx, client in enumerate(client_labels):
        client_counts = np.unique(client, return_counts=True)
        count_idx = 0
        for label in range(10):
            if label in client_counts[0]:
                client_count = client_counts[1][count_idx]
                label_counts[idx].append(client_count)
                if client_count > y_max:
                    y_max = client_count
                count_idx +=1
            else:
                label_counts[idx].append(0)

    colors = ['#c7d200', '#9cac10', '#718620', '#466030', '#23423d', '#00234a', '#063f66', '#0c5e85',
              '#127da4', '#189bc3', '#1fbde6']

    labels = [str(i) for i in range(10)]
    x_ticks = np.arange(len(labels))
    fig, ax = plt.subplots(rows, cols, figsize=(16, 4.5), squeeze=False)
    client = 0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].bar(x_ticks, label_counts[client], color=colors)
            ax[row, col].set_xticks(x_ticks, labels)
            ax[row, col].set_xlabel("Class Label", fontsize=16)
            if col == 0:
                ax[row, col].set_ylabel("Number of Images", fontsize=16)
            if col > 0:
                ax[row, col].set_yticks([])
            ax[row, col].set_title(f"Client {client+1}", fontsize=18)
            ax[row, col].tick_params(axis='both', which='major', labelsize=12)
            ax[row, col].set_ylim(bottom=0, top=y_max+100)
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

'''split data heterogeneously'''
def hetero_split(images: np.ndarray, labels: np.ndarray, n_clients: int, classes_per_client: int):
    class_labels = np.unique(labels)
    client_images = [[] for _ in range(n_clients)]
    client_labels = [[] for _ in range(n_clients)]
    client_iter = 0

    for class_label in class_labels:
        image_subset = images[labels == class_label]
        subset_split = np.array_split(image_subset, classes_per_client)
        for subset in subset_split:
            client_images[client_iter].extend(subset)
            client_labels[client_iter].extend([class_label]*len(subset))
            client_iter +=1
            if client_iter == n_clients:
                client_iter = 0
    return client_images, client_labels

'''split data heterogeneously for 10 clients'''
def hetero_split_10(images: np.ndarray, labels: np.ndarray, n_clients: int, classes_per_client: int):
    client_images = [[] for _ in range(n_clients)]
    client_labels = [[] for _ in range(n_clients)]
    client_num = [0,1]
    label_splits = [np.array_split(labels[labels==i], classes_per_client) for i in range(10)]
    image_splits = [np.array_split(images[labels==i], classes_per_client) for i in range(10)]
    for client_idx in range(10):
        for num in client_num:
            client_images[client_idx].extend(image_splits[num][0])
            image_splits[num].pop(0)
            client_labels[client_idx].extend(label_splits[num][0])
            label_splits[num].pop(0)
        client_num = [(i+1)%10 for i in client_num]
    return client_images, client_labels

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





