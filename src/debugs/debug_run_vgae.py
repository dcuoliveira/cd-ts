import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

class VGAE(torch.nn.Module):
    def __init__(self, n_input, n_hidden_units, n_output):
        super().__init__()
        self.conv1 = GCNConv(n_input, n_hidden_units)
        self.conv2 = GCNConv(n_hidden_units, n_output)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

class VGAEWrapper():
    def __init__(self, input_size, trial):
        self.model_name = "vgae"
        self.params = {
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'n_hidden_units': trial.suggest_int("n_hidden_units", 10, 100),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam"]),
              'n_input': trial.suggest_int("n_input", data.drop([target_name], axis=1).shape[1], data.drop([target_name], axis=1).shape[1]),
              }
        self.epochs = 100

        self.ModelClass = VGAE(n_input=self.params["n_input"],
                               n_hidden_units=self.params["n_hidden_units"],
                               n_output=self.params["n_output"])
        
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

    return roc_auc_score(edge_label.cpu().numpy(), out.cpu().numpy())

def train_and_evaluate_link_prediction(data, model_wrapper, criterion, verbose, trial):

    # instantiate model wrapper
    model_wrapper = model_wrapper(input_size=data.x.shape[1], trial=trial)
    
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
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)

        if verbose:
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
    
    test_auc = eval_link_predictor(model, test_data)

    return test_auc

def objective(data, model_wrapper, criterion, verbose, trial):
         
     loss = train_and_evaluate_link_prediction(data=data,
                                               model_wrapper=model_wrapper,
                                               criterion=criterion,
                                               verbose=verbose,
                                               trial=trial)

     return loss

if __name__ == "__main__":
    import optuna
    import numpy as np
    import os
    import pandas as pd
    from torch_geometric.data import Data

    ## hyperparameters ##
    n_trials = 2
    model_wrapper = VGAEWrapper
    criterion = torch.nn.BCEWithLogitsLoss()
    verbose = False

    ## dataset ##
    source_code = os.path.join(os.getcwd())
    data_files = os.path.join(source_code, "data")

    dgp_simulations = os.listdir(os.path.join(data_files, "simulation"))

    for dgp_name in dgp_simulations:
        dgp_name_summary = "_".join(dgp_name.split(".")[0].split("_")[:2])

        # load asset returns data
        rets_features = pd.read_csv(os.path.join(data_files, "simulation", dgp_simulations[0]), sep=",")
        rets_features.columns = list(range(0, rets_features.shape[1]))

        # load model connection parameters (e.g. B matrix for VAR like models)
        dgp = pd.read_csv(os.path.join(data_files, "DGP", "{}_B.csv".format(dgp_name_summary)), sep=",")
        dgp.columns = list(range(dgp.shape[0], dgp.shape[1] + dgp.shape[0]))

        # infer autoregressive terms and expand returns data
        p = int((dgp.shape[1] / rets_features.shape[1]))
        k = rets_features.shape[1]
        count = k
        for i in range(1, p + 1):
            for j in range(k):
                rets_features[count] = rets_features[j].shift(i)
                count += 1
        rets_features = rets_features.dropna()

        # connection parameters are edges
        edges = dgp.copy()

        # create adjacency matrix from edges
        adj_matrix = np.where(edges.__abs__() != 0, 1, np.nan)
        adj = pd.DataFrame(adj_matrix, index=edges.index, columns=edges.columns).reset_index().melt("index").dropna()

        # create edge index
        row = torch.from_numpy(adj.index.to_numpy().astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.variable.to_numpy().astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # put graph data together
        data = Data(x=x, y=y, edge_index=edge_index, number_of_nodes=x.shape[1])

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: objective(

            data=data,
            model_wrapper=model_wrapper,
            criterion=criterion,
            verbose=verbose,
            trial=trial

            ), n_trials=n_trials)