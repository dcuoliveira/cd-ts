import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from torch import Tensor

def generate_square_subsequent_mask(dim1: int, dim2: int, atten_mask: bool) -> Tensor:
    if atten_mask:
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
    else:
        return torch.triu(torch.ones(dim1, dim2), diagonal=1)

def sample_gumbel(logits, tau=1.0):
    gumbel_noise = -torch.log(1e-10 - torch.log(torch.rand_like(logits) + 1e-10))
    y = logits + gumbel_noise
    return torch.softmax(y / tau, axis=-1)

def find_gpu_device():
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    
    return device_name

def save_pkl(data,
             path):

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

def load_pkl(path):

    with open(path, 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
    return data

def expand_melted_df(adj):

    new_adj = adj.copy()
    for i, row in adj.iterrows():

        if row["variable"] == row["index"]:
            continue

        row_to_add = pd.DataFrame([{"index": row["variable"], "variable": str(row["index"]), "value": row["value"]}])
        new_adj = pd.concat([new_adj, row_to_add], axis=0)
    new_adj = new_adj.reset_index(drop=True)

    return new_adj

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - np.log(1/num_edge_types + eps))
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(target.float().data.view_as(preds)).cpu().sum()
    return correct / (target.size(0) * target.size(1))

def expand_edges(edges):
    
    orig_rows = list(edges.index)
    orig_cols = list(edges.columns)

    for i in list(orig_rows):
        edges[i] = 0
    edges = edges.reindex(sorted(edges.columns), axis=1)

    for j in list(orig_cols):
        edges.loc[j] = edges[j]

    return edges.fillna(0)

def get_off_diag_idx(num_atoms):
    return np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms],
    )

def nll_gaussian(preds, target, variance, add_const=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))