import numpy as np
from sklearn.metrics import accuracy_score

def random_class(adj, adj_test, n, val_size):
    adj = adj.reset_index(drop=True)

    val_accuracy_values = []
    for i in range(n):
        idx = np.random.randint(low=adj.index.min(), high=adj.index.max() + 1, size=int(val_size * adj.shape[0]))
        rand_adj = adj.loc[idx]
        rand_adj["predict"] = np.random.randint(low=0, high=1 + 1, size=int(rand_adj.shape[0]))

        val_accuracy = accuracy_score(y_true=rand_adj["value"], y_pred=rand_adj["predict"])
        val_accuracy_values.append(val_accuracy)
    
    adj_test["predict"] = np.random.randint(low=0, high=1 + 1, size=int(adj_test.shape[0]))
    test_accuracy = accuracy_score(y_true=adj_test["value"], y_pred=adj_test["predict"])

    return test_accuracy, val_accuracy_values 

def majority_class(adj, adj_test, n, val_size):
    adj = adj.reset_index(drop=True)

    val_accuracy_values = []
    for i in range(n):
        idx = np.random.randint(low=adj.index.min(), high=adj.index.max() + 1, size=int(val_size * adj.shape[0]))
        rand_adj = adj.loc[idx]
        rand_adj["predict"] = 0

        val_accuracy = accuracy_score(y_true=rand_adj["value"], y_pred=rand_adj["predict"])
        val_accuracy_values.append(val_accuracy)
    
    adj_test["predict"] = 0
    test_accuracy = accuracy_score(y_true=adj_test["value"], y_pred=adj_test["predict"])

    return test_accuracy, val_accuracy_values 

class RadomClass():
    def __init__(self):
        self.model_name = "random"
        self.model = random_class
        self.n_epochs = 100

class MajorityClass():
    def __init__(self):
        self.model_name = "majority"
        self.model = majority_class
        self.n_epochs = 100
