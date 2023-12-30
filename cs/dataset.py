import scipy.io as sio
import numpy as np
import torch
from pathlib import Path

def load_freyfaces(config):    
    batch_size = config["dataset"]["batch_size"]
    file_folder = Path(__file__).parent / "data" / "frey_rawface.mat"
    matfile = sio.loadmat(file_folder)
    faces = matfile['ff'].T.reshape(-1, 28*20).astype(np.float32)

    size = faces.shape[0]
    test_prop = config["dataset"]["test_prop"]
    train_size = int(test_prop * size)
    test_size = size - train_size
    train_set, test_set = torch.utils.data.random_split(faces, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader

def load_dataset(config):
    if config["dataset"]["name"] == "freyfaces":
        return load_freyfaces(config)
    else:
        raise ValueError("Unknown dataset")
