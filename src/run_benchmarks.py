import optuna
import numpy as np
import os
import pandas as pd
import torch
from torch_geometric.data import Data

from src.models.GNNs import VGAEWrapper
from src.training.runners import gnn_link_prediction_objective
from src.utils.Pyutils import save_pkl

## hyperparameters ##
n_trials = 50
model_wrapper = VGAEWrapper
criterion = torch.nn.BCEWithLogitsLoss()
verbose = False

## dataset ##
source_code = os.path.join(os.getcwd(), "src")
data_files = os.path.join(source_code, "data")
output_files = os.path.join(data_files, "outputs")
dgp_simulations = ["americas", "asia_and_pacific", "europe", "mea"]
dgp_models = ["var"]

if __name__ == "__main__":

    ## Run VGAE for link prediction for ech DGP ##
    results = {}
    error_dgps = {}
    for dgp_name in dgp_simulations:
        for model in dgp_models:

            target_dgp_names = [filename for filename in os.listdir(os.path.join(data_files, "simulation")) if filename.startswith(dgp_name)]
            for file in target_dgp_names:
                # load asset returns data
                rets_features = pd.read_csv(os.path.join(data_files, "simulation", file), sep=",")
                rets_features.columns = list(range(0, rets_features.shape[1])) # fix names

                # load model connection parameters (e.g. B matrix for VAR like models)
                dgp = pd.read_csv(os.path.join(data_files, "DGP", "{}_{}_B.csv".format(dgp_name, model)), sep=",")
                dgp.columns = list(range(dgp.shape[0], dgp.shape[1] + dgp.shape[0])) # fix

                # infer autoregressive terms and expand returns data
                p = int((dgp.shape[1] / rets_features.shape[1]))
                k = rets_features.shape[1]
                count = k

                # expand nodes to include lags
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

                # create features
                x = torch.from_numpy(rets_features.to_numpy()).type(torch.float32)

                # create lables (no lebels for this task)
                y = None

                # put graph data together
                data = Data(x=x, y=y, edge_index=edge_index, number_of_nodes=x.shape[1])

                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
                study.optimize(lambda trial: gnn_link_prediction_objective(

                    data=data,
                    model_wrapper=model_wrapper,
                    criterion=criterion,
                    verbose=verbose,
                    trial=trial

                    ), n_trials=n_trials)
                
                results[file.split(".")[0]] = {
                    
                    "best_params": study.best_params,
                    "train_loss_values": study.best_trial.user_attrs["train_loss_values"],
                    "val_auc_values": study.best_trial.user_attrs["val_auc_values"],
                    "test_auc": study.best_trial.user_attrs["test_auc"]
                    
                    }
        
    save_pkl(data=results,
             path=os.path.join(output_files,  "{}_results.pickle".format("_".join(dgp_models))))
    
    save_pkl(data=error_dgps,
             path=os.path.join(output_files, "{}_error.pickle".format("_".join(dgp_models))))