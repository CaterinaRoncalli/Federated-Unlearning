import sklearn.metrics
import timm
import torch
import torch.nn as nn
import torchvision
from torch import optim

from gym import Gym

train_set = torchvision.datasets.MNIST('files', train=True, download=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=6, persistent_workers=True)

test_set = torchvision.datasets.MNIST('files', train=False, download=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=6, persistent_workers=True)

model = timm.models.xcit_nano_12_p16_224_dist(pretrained=True, in_chans=1, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

gym = Gym(train_loader=train_loader, val_loader=test_loader, model=model, criterion=criterion, optimizer=optimizer,
          device="cpu", verbose=True, metric=sklearn.metrics.balanced_accuracy_score)
gym.train(100, 100)

print("hic featurecloudatum est")
