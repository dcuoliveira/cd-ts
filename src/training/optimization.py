import torch
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T    

import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm  
import numpy as np

@torch.no_grad()
def eval_link_predictor(model, data):
    
    model.eval()
    z = model.encode(x=data.x, edge_index=data.edge_index)

    # sampling training negatives for every training epoch
    neg_edge_index = negative_sampling(edge_index=data.edge_index,
                                        num_nodes=data.num_nodes,
                                        num_neg_samples=data.edge_label_index.size(1), 
                                        method='sparse')

    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z=z, edge_label_index=edge_label_index).view(-1).sigmoid()

    return accuracy_score(edge_label.cpu().numpy(), np.where(out.cpu().numpy() > 0.5, 1, 0))

def train_and_evaluate_link_prediction(data, model_wrapper, criterion, verbose, trial):

    # instantiate model wrapper
    model_wrapper = model_wrapper(input_size=data.number_of_nodes, trial=trial)
    
    # get wrapper parameters
    model = model_wrapper.ModelClass
    param = model_wrapper.params
    n_epochs = model_wrapper.n_epochs
    optimizer = getattr(torch.optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])

    # split links
    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
    )

    train_data, val_data, test_data = split(data)

    train_loss_values = []
    val_auc_values = []
    for epoch in tqdm(range(n_epochs), total=n_epochs + 1, desc="Running backpropagation", disable=not verbose):

        model.train()
        optimizer.zero_grad()
        z = model.encode(x=train_data.x, edge_index=train_data.edge_index)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(edge_index=train_data.edge_index,
                                           num_nodes=train_data.num_nodes,
                                           num_neg_samples=train_data.edge_label_index.size(1), 
                                           method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        train_loss_values.append(loss.item())

        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)
        val_auc_values.append(val_auc)

        if verbose:
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
    train_loss_values_df = pd.DataFrame(train_loss_values, columns=["loss"])
    val_auc_values_df = pd.DataFrame(val_auc_values, columns=["auc"])

    test_auc = eval_link_predictor(model, test_data)

    trial.set_user_attr("train_loss_values", train_loss_values_df) 
    trial.set_user_attr("val_auc_values", val_auc_values_df) 
    trial.set_user_attr("test_auc", test_auc) 

    return test_auc