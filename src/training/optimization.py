import torch
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T    

import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm  
import numpy as np
from sklearn.model_selection import train_test_split

from utils.Pyutils import my_softmax, kl_categorical_uniform, edge_accuracy

def train_and_evaluate_link_prediction_nri(data, target, rel_rec, rel_send, model_wrapper, verbose):
    
    # get wrapper parameters
    model = model_wrapper.ModelClass

    n_epochs = model_wrapper.n_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_auc_values = []
    for epoch in tqdm(range(n_epochs), total=n_epochs + 1, desc="Running backpropagation: train data", disable=not verbose):
        
        for batch_idx, (data, relations) in enumerate(data["train"]):    
            model.train()
            optimizer.zero_grad()
            logits = model.forward(inputs=data, rel_rec=rel_rec, rel_send=rel_send)
            prob = my_softmax(logits, -1)

            loss = kl_categorical_uniform(preds=prob,
                                          num_atoms=data.shape[0],
                                          num_edge_types=2)

            acc = edge_accuracy(preds=logits,
                                target=target)
            train_auc_values.append(acc)

            loss.backward()
            optimizer.step()
    
    test_auc_values = []
    for epoch in tqdm(range(n_epochs), total=n_epochs + 1, desc="Running backpropagation: test data", disable=not verbose):
        
        for batch_idx, (data, relations) in enumerate(data["test"]):    
            model.train()
            optimizer.zero_grad()
            logits = model.forward(inputs=data, rel_rec=rel_rec, rel_send=rel_send)
            prob = my_softmax(logits, -1)

            loss = kl_categorical_uniform(preds=prob,
                                        num_atoms=data.shape[0],
                                        num_edge_types=2)

            acc = edge_accuracy(preds=logits,
                                target=target)
            test_auc_values.append(acc)

            loss.backward()
            optimizer.step()



    return None