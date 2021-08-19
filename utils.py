import random
import pickle
import numpy as np
import torch
import torchvision
from torch.nn.functional import interpolate, grid_sample
import matplotlib.pyplot as plt


def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def interpolate3D(data, shape, mode='bilinear', align_corners=False):
    d_1 = torch.linspace(-1, 1, shape[0])
    d_2 = torch.linspace(-1, 1, shape[1])
    d_3 = torch.linspace(-1, 1, shape[2])
    meshz, meshy, meshx = torch.meshgrid((d_1, d_2, d_3))
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0).to(data.device)

    scaled = grid_sample(data, grid, mode=mode, align_corners=align_corners)
    return scaled


def save_pkl(obj, name, prepath='output/'):
    with open(prepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(name, prepath='output/'):
    with open(prepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)
