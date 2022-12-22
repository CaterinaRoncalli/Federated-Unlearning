from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from mnist import MNIST
import cv2
#import via opencv-python

# nur für eine klasse poisonen --> nur für einen Client
class MNISTDataSet(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, backdoor: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.backdoor = backdoor
        if transform:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize((224, 224)),
                                                  transforms.Normalize((0.1307,), (0.3081,))])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        images = self.transforms(self.images)
        return images[item], self.labels[item]

    def __backdoor__(self, change):
        images = self.images
        labels = self.labels
        backdoorimages = np.empty
        backdoorlabels = np.empty
        for image, label in zip(images, labels):
            if label == change:
                backdoorimage = image.deepcopy()
                backdoorimage = cv2.rectangle(backdoorimage, (24, 24), (25, 25), 250, -5)
                backdoorimage = cv2.rectangle(backdoorimage, (1, 1), (2, 2), 250, -5)
                backdoorlabel = (label + 1) % 10
                np.append(backdoorimages, backdoorimage)
                np.append(backdoorlabels, backdoorlabel)
        images = np.concatenate(images, backdoorimages)
        labels = np.concatenate(labels, backdoorlabels)
        return images, labels




if __name__ == "__main__":

    def client_split(images: np.ndarray, labels: np.ndarray, n_clients: int):
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        splits = np.split(indices, n_clients)
        client_images = []
        client_labels = []
        for split in splits:
            client_images.append(images[split])
            client_labels.append(labels[split])
        return images, labels


    def build_client_loaders(client_images: np.ndarray, client_labels: np.ndarray):
        client_loaders = []
        for images, labels in zip(client_images, client_labels):
            client_set = MNISTDataSet(images, labels)
            client_loader = DataLoader(client_set)
            client_loaders.append(client_loader)
        return client_loaders


    n_clients = 5
    mnist_data = MNIST('files/MNIST/raw/')
    images, labels = mnist_data.load_training()
    split = train_test_split(images, labels, stratify=labels, train_size=0.7)
    train_images, val_images, train_labels, val_labels = split
    test_images, test_labels = mnist_data.load_testing()
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # split MNIST into n clients
    client_train_images, client_train_labels = client_split(train_images, train_labels, n_clients)

    # build list with client loaders
    client_train_loaders = build_client_loaders(client_train_images, client_train_labels)