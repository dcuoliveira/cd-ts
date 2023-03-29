import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset

class LinearDataLoader(Dataset):
    "Simple DataLoader class"

    def __init__(self, root_dir, stateless=False, num_states=1, num_atoms=3, hidden_states=False):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.num_states = num_states
        self.num_atoms = num_atoms
        self.hidden_states = hidden_states

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        filename = self.file_list[i]
        sample = np.load(os.path.join(self.root_dir, filename))
        seq = sample['evolution'].transpose((1,0,2))
        state = seq[:,:,-1:]
        seq = seq[:,:,:-1]
        if not self.hidden_states:
            shape = list(state.shape)
            shape[-1] = self.num_states
            state_onehot = torch.zeros(shape)
            state_onehot.scatter_(-1,torch.from_numpy(state).long(),torch.ones(state.shape))
            feat = np.concatenate([seq, state_onehot.numpy()], axis=-1)
        else:
            feat = seq
        
        off_diag_idx = get_off_diag_idx(self.num_atoms)
        graph = sample['graph'].reshape(self.num_states, self.num_atoms ** 2)[:,off_diag_idx]
        
        return feat, state, graph

def load_springs_data(root_dir, suffix, num_atoms, num_states=2, one_hot=False, hidden_states=False, split='train'):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    loc_train = np.load(os.path.join(root_dir, "loc_" + split + suffix + ".npy"))
    vel_train = np.load(os.path.join(root_dir, "vel_" + split + suffix + ".npy"))
    state_train = np.load(os.path.join(root_dir, "state_" + split + suffix + ".npy"))
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
        state_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        num_atoms,
        num_states,
        one_hot,
        hidden_states
    )

    return train_data


def get_off_diag_idx(num_atoms):
    return np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms],
    )

def data_preparation(
    loc,
    vel,
    state,
    edges,
    loc_min,
    loc_max,
    vel_min,
    vel_max,
    off_diag_idx,
    num_atoms,
    num_states,
    one_hot,
    hidden_states
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Normalize to [-1, 1]
    loc = normalize(loc, loc_min, loc_max)
    vel = normalize(vel, vel_min, vel_max)

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc = np.transpose(loc, [0, 3, 1, 2])
    vel = np.transpose(vel, [0, 3, 1, 2])
    state = np.transpose(state[:,:,None,:], [0, 3, 1, 2])
    if one_hot and not hidden_states:
        shape = list(state.shape)
        shape[-1] = num_states
        state_onehot = torch.zeros(shape)
        state_onehot.scatter_(-1,torch.from_numpy(state).long(),torch.ones(state.shape))
        feat = np.concatenate([loc, vel, state_onehot.numpy()], axis=3)
    elif not hidden_states:
        feat = np.concatenate([loc, vel, state], axis=3)
    else:
        feat = np.concatenate([loc, vel], axis=3)
    print(num_atoms)
    edges = np.reshape(edges, [-1, num_states, num_atoms ** 2])
    edges = np.array((edges + 1) / 2, dtype=np.int64)
    feat = torch.FloatTensor(feat)
    state = torch.LongTensor(state)
    edges = torch.LongTensor(edges)
    edges = edges[:, :, off_diag_idx]
    dataset = TensorDataset(feat, state, edges)

    return dataset

def normalize(x, x_min, x_max):
    return (x - x_min) * 2 / (x_max - x_min) - 1
        


if __name__ == "__main__":


    trainDS = load_springs_data('datasets/data', '_springs5', 5)
    train = torch.utils.data.DataLoader(trainDS, shuffle=True, batch_size=16)
    sample, edges = next(iter(train))
    print(sample.size())
    print(edges.size())