import torch
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T    

import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm  
import numpy as np
from sklearn.model_selection import train_test_split

from utils.Pyutils import my_softmax, kl_categorical_uniform, edge_accuracy

def train_and_evaluate_link_prediction_nri(data, rel_rec, rel_send, model_wrapper, verbose):
    
    # get wrapper parameters
    model = model_wrapper.ModelClass

    n_epochs = model_wrapper.n_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_auc_values = []
    train_loss_values = []
    for epoch in tqdm(range(n_epochs), total=n_epochs + 1, desc="Running backpropagation: train data", disable=not verbose):
        
        for batch_idx, (inputs, target) in enumerate(data["train"]):    
            model.train()
            optimizer.zero_grad()
            logits = model.forward(inputs=inputs, rel_rec=rel_rec, rel_send=rel_send)
            prob = my_softmax(logits, -1)
            # TODO
            # Gumbel-Softmax sampling
            # edges = sample_gumbel(logits, temperature=0.5)
            # Decoding step
            # output = decoder(inputs, edges, rel_rec=rel_rec, rel_send=rel_send)
            # decode_target = inputs[:,:,1:], take T-1 last values

            loss = kl_categorical_uniform(preds=prob,
                                          num_atoms=inputs.shape[1],
                                          num_edge_types=2)
            # loss += nll_likelihood(output, decode_target, var=5e-4)
            train_loss_values.append(loss)

            acc = edge_accuracy(preds=logits,
                                target=target)
            train_auc_values.append(acc)

            loss.backward()
            optimizer.step()
    
    test_auc_values = []
    test_loss_values = []
    for epoch in tqdm(range(n_epochs), total=n_epochs + 1, desc="Running backpropagation: test data", disable=not verbose):
        
        for batch_idx, (inputs, target) in enumerate(data["test"]):    
            model.train()
            optimizer.zero_grad()
            logits = model.forward(inputs=inputs, rel_rec=rel_rec, rel_send=rel_send)
            prob = my_softmax(logits, -1)

            loss = kl_categorical_uniform(preds=prob,
                                        num_atoms=inputs.shape[0],
                                        num_edge_types=2)
            test_loss_values.append(loss)

            acc = edge_accuracy(preds=logits,
                                target=target)
            test_auc_values.append(acc)

            loss.backward()
            optimizer.step()

    results = {
        "train": {"auc": train_auc_values, "loss": train_loss_values},
        "test": {"auc": test_auc_values, "loss": test_loss_values}
               }

    return results