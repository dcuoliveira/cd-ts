import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

from training.optimization import train_and_evaluate_link_prediction_nri
from utils.Pyutils import save_pkl, encode_onehot, expand_edges, get_off_diag_idx

def run_training_procedure(files,
                           input_path,
                           output_path,
                           model_wrapper,
                           batch_size,
                           model_name=None,
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
               continue

          # fix SEM equation names
          dgp.columns = [i + k for i in list(dgp.columns)]
          
          # connection parameters are edges
          edges = dgp.copy()

          # expand edges
          edges = expand_edges(edges)

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
               adj_matrix = torch.FloatTensor(adj_matrix)
               adj_matrix = adj_matrix.view(1, adj_matrix.shape[0], adj_matrix.shape[1])

               # create features
               x = torch.from_numpy(rets_features.to_numpy()).type(torch.float32)

               # generate off-diagonal interaction (fully connected) graph
               # off_diag = dgp.copy()
               # off_diag.loc[:] = 1
               # off_diag = expand_edges(off_diag)
               off_diag = np.ones((adj_matrix.shape[1], adj_matrix.shape[2])) - np.eye(adj_matrix.shape[1])
               rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
               rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
               rel_rec = torch.FloatTensor(rel_rec)
               rel_send = torch.FloatTensor(rel_send)

               # NOTE - we must guarantee that all data have the same number of rows (input size of the MLP) ...
               input_size = x.shape[0] // 2
               train_data = x[0:input_size, ]
               val_data = x[input_size:(input_size * 2), ]

               off_diag_idx = get_off_diag_idx(adj_matrix.shape[1])
               tensor_edges = torch.reshape(adj_matrix, [-1, adj_matrix.shape[1] ** 2])
               tensor_edges = (tensor_edges + 1) // 2
               tensor_edges = tensor_edges[:, off_diag_idx]

               # create dataloader
               train_data = train_data.T.view(1, train_data.T.shape[0], train_data.T.shape[1])
               test_data = val_data.T.view(1, val_data.T.shape[0], val_data.T.shape[1])
               
               tensor_train_data = TensorDataset(train_data, tensor_edges)
               tensor_test_data = TensorDataset(test_data, tensor_edges)

               train_data_loader = DataLoader(tensor_train_data, batch_size=batch_size, shuffle=True, num_workers=8)
               test_data_loader = DataLoader(tensor_test_data, batch_size=batch_size, shuffle=True, num_workers=8)

               data_loader = {

                    "train": train_data_loader,
                    "test": test_data_loader
                              
                              }

               # NOTE - n_in = number of observations in the paper ...
               model_wrapper = model_wrapper(n_in=input_size)

               results = train_and_evaluate_link_prediction_nri(data=data_loader,
                                                                rel_rec=rel_rec,
                                                                rel_send=rel_send,
                                                                model_wrapper=model_wrapper,
                                                                verbose=verbose)

          save_pkl(data=results, path=os.path.join(output_path,  "{}_results.pickle".format(file.split(".")[-2])))

