import optuna
import numpy as np
import os
import pandas as pd
import torch
from torch_geometric.data import Data

from models.GNNs import VGAEWrapper
from training.runners import gnn_link_prediction_objective
from utils.Pyutils import save_pkl

## hyperparameters ##
n_trials = 50
model_wrapper = VGAEWrapper
criterion = torch.nn.BCEWithLogitsLoss()
verbose = False

## dataset path ##
source_code = os.path.join(os.getcwd(), "src")
data_files = os.path.join(source_code, "data")
output_files = os.path.join(data_files, "outputs")

## multivariate time-series process characteristics ##
dgp_simulations = ["americas", "asia_and_pacific", "europe", "mea"]
Ts = [100, 500, 1000, 2000, 3000, 4000, 5000]
functional_forms = ["linear", "nonlinear"]
error_term_dists = ["gaussian", "nongaussian"]
sampling_freq = ["daily", "monthly"]

if __name__ == "__main__":

    ## Run VGAE for link prediction for ech DGP ##
    results = {}
    error_dgps = {}
    for g in dgp_simulations:
            for T in Ts:
                 for f in functional_forms:
                      for e in error_term_dists:
                           for freq in sampling_freq:
                                file_name = "{g}_{T}_{f}_{e}_{freq}.npz".format(g=g, T=T, f=f, e=e, freq=freq)

                                # load simulation
                                sim_data = np.load(file=os.path.join(data_files, "simulations", file_name))

                                # load asset returns data
                                rets_features = pd.DataFrame(sim_data["simulation"])

                                # load model connection parameters (e.g. B matrix for VAR like models)
                                dgp = pd.DataFrame(sim_data["coefficients"])

                                # infer autoregressive terms and expand returns data
                                p = int((dgp.shape[1] / rets_features.shape[1]))
                                k = rets_features.shape[1]
                                count = k

                                # expand nodes to include lags - assumption: p is known
                                old_rets_features = rets_features.copy()
                                for i in range(1, p + 1):
                                    for j in range(k):
                                        rets_features[count] = rets_features[j].shift(i)
                                        count += 1
                                rets_features = rets_features.dropna()

                                # fix SEM equation names
                                dgp.columns = [i + k for i in list(dgp.columns)]

                                # assumption: no comtemporaneous effects - select lagged variables only
                                rets_features = rets_features[dgp.columns]
                                
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