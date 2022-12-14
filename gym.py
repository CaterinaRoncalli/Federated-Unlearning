from copy import deepcopy
import numpy as np
import torch.optim.optimizer
from sklearn import metrics
from torch import nn, Tensor, inference_mode
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class Gym:
    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 scheduler: object = None,
                 device: str | int = 'cuda',
                 metric: callable = metrics.balanced_accuracy_score,
                 verbose: bool = True):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.metric = metric
        self.verbose = verbose

    def train(self, epochs: int, eval_interval: int = 10) -> nn.Module:
        epochs = range(epochs)
        if self.verbose:
            epoch_bar = tqdm(epochs, total=len(epochs) * len(self.train_loader))
        iterations = 0
        for _ in epochs:
            for train_data in self.train_loader:
                inputs, labels = train_data
                loss = self._train_batch(inputs=inputs, labels=labels)
                if iterations % eval_interval == 0:
                    metric = self.eval()
                if self.verbose:
                    epoch_bar.update(1)
                    epoch_bar.set_description(f'loss: {loss:.4f}, metric value: {metric:.4f}')
                iterations += 1
        return deepcopy(self.model)

    @autocast()
    def _train_batch(self, inputs: Tensor, labels: Tensor) -> float:
        self.model.train()
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    @autocast()
    @inference_mode()
    def eval(self) -> float:
        loss_list = []
        output_list = []
        label_list = []
        self.model.eval()
        for idx, (inputs, labels) in enumerate(self.val_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())
            output_list.extend(outputs.detach().tolist())
            label_list.extend(labels.detach().tolist())
        pred = np.argmax(np.array(output_list), axis=1)
        metric = self.metric(np.array(label_list), pred)
        return metric
