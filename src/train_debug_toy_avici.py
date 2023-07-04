import os
import torch
import numpy as np
from tqdm import tqdm

from models.TransformerEncoder import TransformerEncoder
from data_loaders import load_springs_data
from utils.Pyutils import find_gpu_device, encode_onehot

# network hyperparameters
num_atoms = 5
dim_val=512
n_encoder_layers=4
n_heads=8
dropout_encoder=0.2
dropout_pos_enc=0.1
dim_feedforward_encoder=2048

# optimizer hyperparameters
n_epochs = 500
learning_rate = 5e-4
learning_rate_decay_step = 200
gamma = 0.5

# running parameters
device = torch.device(find_gpu_device())
dataset_name = "spring_data_test"
file_name = "_springs{}".format(num_atoms) # NOTE: changed to 1000 instead of 10000 (missing edges file for 10000)
root_path = os.path.dirname(__file__)
data_path = os.path.join(root_path, "data")

## load data
train_dataset = load_springs_data(root_dir=os.path.join(data_path, dataset_name),
                                  suffix=file_name,
                                  num_atoms=num_atoms)

# NOTE: Random sampling occurs in the "num_sample" dimension instead of "num_obs"
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
off_diag = np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec).to(device)
rel_send = torch.FloatTensor(rel_send).to(device)

# load Models
encoder = TransformerEncoder(input_size=num_atoms,
                             dim_val=dim_val,  
                             n_encoder_layers=n_encoder_layers,
                             n_heads=n_heads,
                             dropout_encoder=dropout_encoder, 
                             dropout_pos_enc=dropout_pos_enc,
                             dim_feedforward_encoder=dim_feedforward_encoder,
                             num_predicted_features=num_atoms).to(device)

# optimizers
optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_step, gamma=gamma)

# train
encoder.train()
for i in range(n_epochs):

    print("Epoch:", i)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (features, gt_edges) in pbar:
        B = features.shape[0]
        optimizer.zero_grad()
        features = features.to(device)