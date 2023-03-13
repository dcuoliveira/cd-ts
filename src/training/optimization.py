import torch
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T    

import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm  
import numpy as np
from sklearn.model_selection import train_test_split

from utils.Pyutils import my_softmax, kl_categorical_uniform, edge_accuracy

def train_and_evaluate_link_prediction_nri(data, target, rel_rec, rel_send, model_wrapper, verbose, trial):

    # TODO - n_in = number of observations in the paper ...
    # instantiate model wrapper
    model_wrapper = model_wrapper(n_in=data.shape[1], trial=trial)
    
    # get wrapper parameters
    model = model_wrapper.ModelClass

    # TODO - Optimizing over parameters, the authors avoid thi
    param = model_wrapper.params
    n_epochs = model_wrapper.n_epochs
    optimizer = getattr(torch.optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

    # TODO - Random splits, even for time series
    # TODO - Splits considered at the time dimension
    train_data, val_data = train_test_split(data, test_size=0.8)
    val_data, test_data = train_test_split(val_data, test_size=0.5)

    train_auc_values = []
    for epoch in tqdm(range(n_epochs), total=n_epochs + 1, desc="Running backpropagation: train data", disable=not verbose):

        model.train()
        optimizer.zero_grad()
        logits = model.forward(inputs=train_data, rel_rec=rel_rec, rel_send=rel_send)
        prob = my_softmax(logits, -1)

        loss = kl_categorical_uniform(preds=prob,
                                      num_atoms=data.shape[0],
                                      num_edge_types=2)

        acc = edge_accuracy(preds=logits,
                            target=target)
        train_auc_values.append(acc)

        loss.backward()
        optimizer.step()

    val_auc_values = []
    for epoch in tqdm(range(n_epochs), total=n_epochs + 1, desc="Running backpropagation: test data", disable=not verbose):

        model.train()
        optimizer.zero_grad()
        logits = model.forward(inputs=val_data, rel_rec=rel_rec, rel_send=rel_send)
        prob = my_softmax(logits, -1)

        loss = kl_categorical_uniform(preds=prob,
                                      num_atoms=data.shape[0],
                                      num_edge_types=2)

        acc = edge_accuracy(preds=logits,
                            target=target)
        val_auc_values.append(acc)

        loss.backward()
        optimizer.step()

    return None