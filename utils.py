from MNIST_Dataset import MNISTDataSet
import numpy as np
from sklearn import metrics
import torch.nn as nn
from torch import inference_mode
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from typing import List


def client_split(images: np.ndarray, labels: np.ndarray, n_clients: int) -> (List[np.ndarray], List[np.ndarray]):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    splits = np.split(indices, n_clients)
    client_images = []
    client_labels = []
    for split in splits:
        client_images.append(images[split])
        client_labels.append(labels[split])
    return client_images, client_labels


def build_client_loaders(client_images: np.ndarray,
                         client_labels: np.ndarray,
                         backdoor: bool,
                         batch_size: int,
                         num_workers: int,
                         shuffle: bool,
                         persistent_workers: bool,
                         backdoor_old_label: int | None = None,
                         backdoor_new_label: int | None = None) -> List[DataLoader]:
    client_loaders = []
    for images, labels in zip(client_images, client_labels):
        client_set = MNISTDataSet(images, labels, backdoor=backdoor, backdoor_old_label=backdoor_old_label,
                                  backdoor_new_label=backdoor_new_label)
        client_loader = DataLoader(client_set, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, persistent_workers=persistent_workers)
        client_loaders.append(client_loader)
    return client_loaders


@autocast()
@inference_mode()
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
