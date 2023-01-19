from copy import deepcopy
import numpy as np
from sklearn import metrics
from torch import nn, Tensor, inference_mode
from torch.cuda.amp import autocast
import torch.optim.optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Callable, Tuple
import wandb


class Gym:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader | None = None,
                 scheduler: object = None,
                 device: str | int = 'cuda',
                 metric: Callable | None = None,
                 verbose: bool = True,
                 name: str | int | None = None,
                 log: bool = False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.metric = metric
        self.scaler = torch.cuda.amp.GradScaler()
        self.verbose = verbose
        self.name = name
        self.log = log

    def train(self, epochs: int, eval_interval: int = 10) -> nn.Module:
        best_model = self.model
        best_metric = 0
        epochs = range(epochs)
        if self.verbose:
            epoch_bar = tqdm(epochs, total=len(epochs) * len(self.train_loader))
        iterations = 0
        metric = np.nan
        for _ in epochs:
            for train_data in self.train_loader:
                inputs, labels = train_data
                loss = self._train_batch(inputs=inputs, labels=labels)
                if self.log:
                    wandb.log({f'train loss/{self.name}': loss})
                if iterations % eval_interval == 0 and self.val_loader:
                    metric = self.eval()
                    if self.scheduler:
                        self.scheduler.step(metric)
                    if best_metric < metric:
                        best_metric = metric
                        best_model = deepcopy(self.model)
                    if self.log:
                        wandb.log({f'{str(self.metric)}/{self.name}': metric})
                if self.verbose:
                    epoch_bar.update(1)
                    epoch_bar.set_description(f'{self.name} train loss: {loss:.4f}, metric value: {metric:.4f}')
                iterations += 1
        return deepcopy(best_model)

    @autocast()
    def _train_batch(self, inputs: Tensor, labels: Tensor) -> float:
        self.model.train()
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    @autocast()
    @inference_mode()
    def eval(self) -> float:
        output_list = []
        label_list = []
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        for idx, (inputs, labels) in enumerate(self.val_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            output_list.extend(outputs.detach().tolist())
            label_list.extend(labels.detach().tolist())
        pred = np.argmax(np.array(output_list), axis=1)
        metric = self.metric(np.array(label_list), pred)
        return metric


class FederatedGym:
    def __init__(self,
                 client_train_loaders: List[DataLoader],
                 val_loader: DataLoader,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion: nn.Module,
                 optimizer_params: dict | None = None,
                 scheduler: object = None,
                 device: str | int = 'cuda',
                 metric: callable = metrics.balanced_accuracy_score,
                 verbose: bool = True,
                 log: bool = True):
        self.client_train_loaders = client_train_loaders
        self.val_loader = val_loader
        self.global_model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.optimizer_params = optimizer_params or dict()
        self.scheduler = scheduler
        self.device = device
        self.metric = metric
        self.verbose = verbose
        self.log = log

    def train(self, epochs: int, rounds: int) -> Tuple[nn.Module, List[nn.Module]]:
        best_model = deepcopy(self.global_model)
        best_client_models = None
        best_metric = -1
        for train_round in range(rounds):
            if self.verbose:
                print(f"Round nr: {train_round}")
            client_models = self.train_clients(epochs)
            self._aggregate_models(client_models)
            self.global_model.to(device=self.device)
            metric = self.eval_global_model()
            if best_metric < metric:
                best_model = deepcopy(self.global_model)
                best_client_models = client_models
                best_metric = metric
            if self.log:
                wandb.log({f'{str(self.metric)}/global': metric})
            # self.global_model.cpu()
            if self.verbose:
                print(f"metric value: {metric:.4f} for round nr {train_round}")
            del client_models
        return best_model, best_client_models

    def train_clients(self, epochs: int) -> List[nn.Module]:
        client_models = []
        for client_number, client_train_loader in enumerate(self.client_train_loaders):
            client_gym = self._init_client(train_loader=client_train_loader,
                                           model=self.global_model, client_number=client_number)
            client_model = client_gym.train(epochs=epochs)
            client_models.append(client_model.cpu())
        return client_models

    def _init_client(self,
                     train_loader: DataLoader,
                     model: nn.Module, client_number: int) -> Gym:
        client_model = deepcopy(model)
        optimizer = self.optimizer(params=client_model.parameters(), **self.optimizer_params)
        client_gym = Gym(train_loader=train_loader,
                           model=client_model,
                           optimizer=optimizer,
                           criterion=self.criterion,
                           device=self.device,
                           name=f"client number {client_number+1}", log=self.log)
        return client_gym

    def _aggregate_models(self, client_models: List[nn.Module]):
        client_parameters = [model.parameters() for model in client_models]
        weights = torch.as_tensor([len(train_loader) for train_loader in self.client_train_loaders])
        weights = weights / weights.sum()
        for model_parameter in zip(self.global_model.parameters(), *client_parameters):
            global_parameter = model_parameter[0]
            client_parameter = [client_parameter.data * weight for client_parameter, weight in
                                 zip(model_parameter[1:], weights)]
            client_parameter = torch.stack(client_parameter, dim=0).sum(dim=0)
            global_parameter.data = client_parameter

    @autocast()
    @inference_mode()
    def eval_global_model(self) -> float:
        output_list = []
        label_list = []
        self.global_model.eval()
        for idx, (inputs, labels) in enumerate(self.val_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.global_model(inputs)
            output_list.extend(outputs.detach().tolist())
            label_list.extend(labels.detach().tolist())
        pred = np.argmax(np.array(output_list), axis=1)
        metric = self.metric(np.array(label_list), pred)
        return metric


class UnlearnGym(Gym):
    def __init__(self, criterion: nn.Module,  *args, **kwargs):
        super().__init__(criterion=criterion, *args, **kwargs)
        self.criterion = lambda *l_args, **l_kwargs: -1 * criterion(*l_args, **l_kwargs)

    def untrain(self, *args, **kwargs):
        return self.train(*args, **kwargs)


class ClientUnlearnGym(UnlearnGym):
    def __init__(self, global_model: nn.Module, n_clients: int, delta: float | None, tau: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.n_clients = n_clients
        self.ref_model = self.calc_ref_model()
        if delta:
            self.deltas = [delta] * sum(1 for _ in self.global_model.parameters()) #radius around ref model
        else:
            self.deltas = self.calc_delta()
        self.tau = tau #early stopping criterion


    def untrain(self, epochs: int, *args, **kwargs):
        for epoch in range(epochs):
            for train_data in self.train_loader:
                inputs, labels = train_data
                loss = self._untrain_batch(inputs=inputs, labels=labels)
                metric = self.eval()
                if metric < self.tau:
                    break
                print(f'unlearned client model metric: {metric}')
            if metric < self.tau:
                print(f'early stopped after {epoch} epochs')
                break
        return deepcopy(self.model)

    @autocast()
    def _untrain_batch(self, inputs: Tensor,  labels: Tensor):
        self.model.train()
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.project_norm_ball()
        self.scaler.update()
        return loss.item()

    def calc_ref_model(self) -> nn.Module:
        ref_model = deepcopy(self.global_model)
        param_iterator = zip(self.global_model.parameters(), self.model.parameters(), ref_model.parameters())
        for global_parameter, client_parameter, ref_parameter in param_iterator:
            diff_parameter = 1 / (self.n_clients - 1) * (self.n_clients * global_parameter - client_parameter)
            ref_parameter.data = diff_parameter
        return ref_model

    def calc_delta(self):
        model = deepcopy(self.global_model)
        random_models_params = []
        deltas = []
        for i_model in range(10):
            model.apply(self.weight_reset)
            for i_param, (ref_params, rand_params) in enumerate(zip(self.ref_model.parameters(), model.parameters())):
                diff = (ref_params - rand_params).detach().cpu().flatten().numpy()
                delta = np.linalg.norm(diff, ord=2)
                if i_model == 0:
                    deltas.append(delta)
                else:
                    deltas[i_param] += delta
        for i_delta, delta in enumerate(deltas):
            deltas[i_delta] = (1 / 3) * delta / 10
        return deltas

    @staticmethod
    def weight_reset(model):
        reset_parameters = getattr(model, "reset_parameters", None)
        if callable(reset_parameters):
            model.reset_parameters()

    def project_norm_ball(self):
        for ref_parameter, client_parameter, delta in zip(self.ref_model.parameters(), self.model.parameters(), self.deltas):
            diff = ref_parameter - client_parameter
            distance = diff.norm(p=2)
            scale_factor = delta / distance
            if scale_factor >= 1:
                pass
            else:
                client_parameter = client_parameter * scale_factor


class FederatedUnlearnGym(FederatedGym):
    def __init__(self,
                 unclient_model: nn.Module,
                 unclient_train_loader: DataLoader,
                 unclient_val_loader: DataLoader,
                 delta: float | None,
                 tau: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unclient_model = unclient_model
        self.unclient_train_loader = unclient_train_loader
        self.unclient_val_loader = unclient_val_loader
        self.delta = delta
        self.tau = tau
        
    def untrain(self,
                untrain_optimizer: torch.optim.Optimizer,
                client_untrain_epochs: int,
                federated_epochs: int, 
                federated_rounds: int) -> Tuple[nn.Module, List[nn.Module], nn.Module]:
        unfed_gym = ClientUnlearnGym(train_loader=self.unclient_train_loader, val_loader=self.unclient_val_loader,
                                     model=self.unclient_model,
                                     global_model=self.global_model, criterion=self.criterion,
                                     optimizer=untrain_optimizer, device=self.device, verbose=True,
                                     metric=self.metric,
                                     delta=self.delta, tau=self.tau, n_clients=len(self.client_train_loaders)+1,
                                     log=self.log)

        self.global_model = unfed_gym.untrain(epochs=client_untrain_epochs)
        unlearned_model = deepcopy(self.global_model)
        global_model, client_models = self.train(epochs=federated_epochs, rounds=federated_rounds)
        return global_model, client_models, unlearned_model

