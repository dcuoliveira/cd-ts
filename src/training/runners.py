import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import optuna
from sklearn.model_selection import train_test_split

from training.optimization import train_and_evaluate_link_prediction_nri
from utils.Pyutils import save_pkl, expand_melted_df, encode_onehot

def run_training_procedure(files,
                           input_path,
                           output_path,
                           model_wrapper,
                           model_name=None,
                           n_trials=None,
                           criterion=None,
                           verbose=False):

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

          if rets_features.shape[0] < rets_features.shape[1]:
               a=1
               continue

          # fix SEM equation names
          dgp.columns = [i + k for i in list(dgp.columns)]

          # assumption: no comtemporaneous effects - select lagged variables only
          rets_features = rets_features[dgp.columns]
          
          # connection parameters are edges
          edges = dgp.copy()

          # create adjacency matrix from edges
          adj_matrix = np.where(edges.__abs__() != 0, 1, 0)
          adj_matrix_df = pd.DataFrame(adj_matrix, index=edges.index, columns=edges.columns)

          adj = adj_matrix_df.reset_index().melt("index")

          if model_name == "random":
               wrapper = model_wrapper()
               adj_train, adj_test = train_test_split(adj, test_size=0.1, random_state=0)

               test_auc, val_auc_values = wrapper.model(adj=adj_train, adj_test=adj_test, n=wrapper.n_epochs, val_size=0.1)

               best_params = np.nan
               train_loss_values = np.nan
          elif model_name == "majority":
               wrapper = model_wrapper()
               adj_train, adj_test = train_test_split(adj, test_size=0.1, random_state=0)

               test_auc, val_auc_values = wrapper.model(adj=adj_train, adj_test=adj_test, n=wrapper.n_epochs, val_size=0.1)

               best_params = np.nan
               train_loss_values = np.nan
          elif model_name == "nrimlp":
               # create target
               adj = adj.loc[adj["value"] != 0]
               adj = expand_melted_df(adj=adj)
               target = torch.from_numpy(adj.pivot_table(index="index", columns="variable", values="value").fillna(0).to_numpy()).type(torch.float32)

               # create features
               x = torch.from_numpy(rets_features.to_numpy()).type(torch.float32)

               # Generate off-diagonal interaction graph
               off_diag = np.ones([rets_features.shape[1], rets_features.shape[1]]) # - np.eye(rets_features.shape[1])
               rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
               rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
               rel_rec = torch.FloatTensor(rel_rec)
               rel_send = torch.FloatTensor(rel_send)

               results = train_and_evaluate_link_prediction_nri(data=x,
                                                                target=target,
                                                                rel_rec=rel_rec,
                                                                rel_send=rel_send,
                                                                model_wrapper=model_wrapper,
                                                                criterion=criterion,
                                                                verbose=verbose)
               
          results[file.split(".")[0]] = {
               
               "best_params": best_params,
               "train_loss_values": train_loss_values,
               "val_auc_values": val_auc_values,
               "test_auc": test_auc
               
               }

          save_pkl(data=results, path=os.path.join(output_path,  "{}_results.pickle".format(file.split(".")[-2])))

