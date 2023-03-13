import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

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
        row_to_add = pd.DataFrame([{"index": row["variable"], "variable": str(row["index"]), "value": 1}])
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
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))