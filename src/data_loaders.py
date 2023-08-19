import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset

def load_economic_simulations(root_dir, suffix, num_atoms, split='train'):

    loc_train = np.load(os.path.join(root_dir, "loc_" + split + suffix + ".npy"))
    vel_train = np.load(os.path.join(root_dir, "vel_" + split + suffix + ".npy"))
    edges_train = np.load(os.path.join(root_dir, "edges_" + split + suffix + ".npy"))
    
    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)

    train_data = data_preparation(
        loc_train,
        vel_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        num_atoms
    )

    return train_data

def load_data(root_dir, suffix, num_atoms, split='train'):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    loc_train = np.load(os.path.join(root_dir, "loc_" + split + suffix + ".npy"), allow_pickle=True)
    vel_train = np.load(os.path.join(root_dir, "vel_" + split + suffix + ".npy"), allow_pickle=True)
    edges_train = np.load(os.path.join(root_dir, "edges_" + split + suffix + ".npy"), allow_pickle=True)

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)

    if (len(vel_train.shape) != 0) and (len(loc_train.shape) != 0):
        loc_max = loc_train.max()
        loc_min = loc_train.min()
        vel_max = vel_train.max()
        vel_min = vel_train.min()

        train_data = data_preparation(
            loc_train,
            vel_train,
            edges_train,
            loc_min,
            loc_max,
            vel_min,
            vel_max,
            off_diag_idx,
            num_atoms
        )
    else:

        if (len(vel_train.shape) != 0):
            final_loc_train = vel_train
        elif (len(loc_train.shape) != 0):
            final_loc_train = loc_train
        else:
            raise ValueError("Both loc_train and vel_train are empty arrays.")

        final_loc_max = loc_train.max()
        final_loc_min = loc_train.min()

        train_data = data_preparation_simple(
            final_loc_train,
            edges_train,
            final_loc_min,
            final_loc_max,
            off_diag_idx,
            num_atoms
        )

    return train_data

def load_springs_data(root_dir, suffix, num_atoms, split='train'):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    loc_train = np.load(os.path.join(root_dir, "loc_" + split + suffix + ".npy"))
    vel_train = np.load(os.path.join(root_dir, "vel_" + split + suffix + ".npy"))
    edges_train = np.load(os.path.join(root_dir, "edges_" + split + suffix + ".npy"))
    
    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)

    train_data = data_preparation(
        loc_train,
        vel_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        num_atoms
    )

    return train_data


def get_off_diag_idx(num_atoms):
    return np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms],
    )

def data_preparation_simple(
    loc,
    edges,
    loc_min,
    loc_max,
    off_diag_idx,
    num_atoms,
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Normalize to [-1, 1]
    loc = normalize(loc, loc_min, loc_max)

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    feat = np.transpose(loc, [0, 3, 1, 2])
    edges = np.reshape(edges, [edges.shape[0], edges.shape[1] * edges.shape[2]])
    edges = np.array((edges + 1) / 2, dtype=np.int64)
    feat = torch.FloatTensor(feat)
    edges = torch.LongTensor(edges)
    edges = edges[:, off_diag_idx]
    dataset = TensorDataset(feat, edges)

    return dataset

def data_preparation(
    loc,
    vel,
    edges,
    loc_min,
    loc_max,
    vel_min,
    vel_max,
    off_diag_idx,
    num_atoms,
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Normalize to [-1, 1]
    loc = normalize(loc, loc_min, loc_max)
    vel = normalize(vel, vel_min, vel_max)

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc = np.transpose(loc, [0, 3, 1, 2])
    vel = np.transpose(vel, [0, 3, 1, 2])
    feat = np.concatenate([loc, vel], axis=3)
    edges = np.reshape(edges, [-1, num_atoms ** 2])
    edges = np.array((edges + 1) / 2, dtype=np.int64)
    feat = torch.FloatTensor(feat)
    edges = torch.LongTensor(edges)
    edges = edges[:, off_diag_idx]
    dataset = TensorDataset(feat, edges)

    return dataset

def normalize(x, x_min, x_max):
    return (x - x_min) * 2 / (x_max - x_min) - 1
        


if __name__ == "__main__":


    trainDS = load_springs_data('datasets/data', '_springs5', 5)
    train = torch.utils.data.DataLoader(trainDS, shuffle=True, batch_size=16)
    sample, edges = next(iter(train))
    print(sample.size())
    print(edges.size())