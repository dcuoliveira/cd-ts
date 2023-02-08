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
              'n_output': trial.suggest_int("n_output", 10, 20),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam"]),
              'n_input': trial.suggest_int("n_input", input_size, input_size),
              }
        self.n_epochs = 100

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
    from utils.Pyutils import save_pkl

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

                try:
                    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
                    study.optimize(lambda trial: objective(

                        data=data,
                        model_wrapper=model_wrapper,
                        criterion=criterion,
                        verbose=verbose,
                        trial=trial

                        ), n_trials=n_trials)
                except:
                    error_dgps[file.split(".")[0]] = {"data": data}

                    continue
                
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