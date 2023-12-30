import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

# read mat file
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import math
from torch.optim import Adam
import torch
import numpy as np

matfile = sio.loadmat('./frey_rawface.mat')
faces = matfile['ff'].T.reshape(-1, 28, 20)
print(matfile.keys())
print(matfile['ff'].shape)
print(faces.shape)
print(faces.dtype)

# show two faces

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

imgplot = plt.imshow(faces[0,:,:], cmap='gray')
plt.show()

imgplot = plt.imshow(faces[1,:,:], cmap='gray')
plt.show()