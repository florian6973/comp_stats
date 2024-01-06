import scipy.io as sio
import numpy as np
import torch
from pathlib import Path
from torchvision.datasets import MNIST, Omniglot, Caltech101, FashionMNIST
from torch.utils.data import DataLoader
# gaussian mixture EM
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import TensorDataset

def load_freyfaces(config):    
    batch_size = config["dataset"]["batch_size"]
    file_folder = Path(__file__).parent / "data" / "frey_rawface.mat"
    matfile = sio.loadmat(file_folder)
    faces = matfile['ff'].T.reshape(-1, 28*20).astype(np.float32)

    data = TensorDataset(torch.FloatTensor(faces), torch.zeros(len(faces)))

#    gmm = GaussianMixture(n_components=10, covariance_type='diag')  

    return split(config, batch_size, data, 28*20)

def load_mnist(config):
    batch_size = config["dataset"]["batch_size"]
    dataset = MNIST("mnist", download=True)
    data = dataset.train_data.float().reshape(-1, 28*28)
    data_labels = dataset.train_labels

    data = (data > 255/2).float()
    data = TensorDataset(data, data_labels)

    # plt.imshow(data[0].reshape(28, 28), cmap='gray')
    # plt.axis('off')
    # plt.show()
    # exit()

    return split(config, batch_size, data, 28*28)

def load_fashion_mnist(config):
    batch_size = config["dataset"]["batch_size"]
    dataset = FashionMNIST("fashion_mnist", download=True)
    data = dataset.train_data.float().reshape(-1, 28*28)
    data_labels = dataset.train_labels

    data = (data > 255/2).float()

    data = TensorDataset(data, data_labels)

    return split(config, batch_size, data, 28*28)

def load_omniglot(config):
    batch_size = config["dataset"]["batch_size"]
    dataset = Omniglot("omniglot", download=True, transform=torchvision.transforms.ToTensor())
    data = torch.zeros((len(dataset), 105, 105))
    labels = torch.zeros((len(dataset), 1))
    for i in range(len(dataset)):
        print('\r', f"Loading {i+1}/{len(dataset)}", end="")
        data[i] = dataset[i][0]
        labels[i] = dataset[i][1]
    print()

    data = (data > 255/2).float()

    return split(config, batch_size, data, 105*105)

def load_silhouette(config):
    batch_size = config["dataset"]["batch_size"]
    file_folder = Caltech101("caltech101", download=True)
    data = file_folder.test_data.float().reshape(-1, 28*28)

    data = (data > 255/2).float()

    return split(config, batch_size, data, 28*28)

def split(config, batch_size, data, dim_input):
    size = len(data)
    train_prop = config["dataset"]["train_prop"]
    train_size = int(train_prop * size)
    test_size = size - train_size
    train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])

    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")
    # print(f"Shape: {data.shape} = {train_set.dataset.data.shape} + {test_set.dataset.data.shape}")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    
    return train_loader,test_loader,dim_input

def load_dataset(config):
    if config["dataset"]["name"] == "freyfaces":
        return load_freyfaces(config)
    elif config["dataset"]["name"] == "mnist":
        return load_mnist(config)
    elif config["dataset"]['name'] == "fashion_mnist":
        return load_fashion_mnist(config)
    elif config["dataset"]["name"] == "omniglot":
        return load_omniglot(config)
    elif config["dataset"]["name"] == "silhouette":
        return load_silhouette(config)
    else:
        raise ValueError("Unknown dataset")
