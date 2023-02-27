import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import optuna
from sklearn.model_selection import train_test_split

from training.optimization import train_and_evaluate_link_prediction
from utils.Pyutils import save_pkl

def gnn_link_prediction_objective(data, model_wrapper, criterion, verbose, trial):
         
     loss = train_and_evaluate_link_prediction(data=data,
                                               model_wrapper=model_wrapper,
                                               criterion=criterion,
                                               verbose=verbose,
                                               trial=trial)

     return loss

def run_training_procedure(files, input_path, output_path, model_wrapper, n_trials=None, criterion=None, verbose=False):

     results = {}
     for file in files:
          
          # load simulation
          sim_data = np.load(file=os.path.join(input_path, file))

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
          adj_matrix = np.where(edges.__abs__() != 0, 1, 0)
          adj = pd.DataFrame(adj_matrix, index=edges.index, columns=edges.columns).reset_index().melt("index")

          wrapper = model_wrapper()
          if wrapper.model_name == "random":
               adj_train, adj_test = train_test_split(adj, test_size=0.1, random_state=0)

               test_auc, val_auc_values = wrapper.model(adj=adj_train, adj_test=adj_test, n=wrapper.n_epochs, val_size=0.1)

               best_params = np.nan
               train_loss_values = np.nan
          elif wrapper.model_name == "majority":
               adj_train, adj_test = train_test_split(adj, test_size=0.1, random_state=0)

               test_auc, val_auc_values = wrapper.model(adj=adj_train, adj_test=adj_test, n=wrapper.n_epochs, val_size=0.1)

               best_params = np.nan
               train_loss_values = np.nan
          elif wrapper.model_name == "vgae":
               # delete no edge rows

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
               
               best_params = study.best_params
               train_loss_values = study.best_trial.user_attrs["train_loss_values"]
               val_auc_values = study.best_trial.user_attrs["val_auc_values"]
               test_auc = study.best_trial.user_attrs["test_auc"]
               
          results[file.split(".")[0]] = {
               
               "best_params": best_params,
               "train_loss_values": train_loss_values,
               "val_auc_values": val_auc_values,
               "test_auc": test_auc
               
               }

          save_pkl(data=results, path=os.path.join(output_path,  "{}_results.pickle".format(file.split(".")[-2])))

