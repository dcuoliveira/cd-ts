import os
import torch
import numpy as np
from tqdm import tqdm

from models.TransformerEncoder import TransformerEncoder
from data_loaders import load_springs_data
from utils.Pyutils import find_gpu_device, encode_onehot

# running parameters
device = torch.device(find_gpu_device())
num_atoms = 5
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
encoder = TransformerEncoder(input_size=1,
                             batch_first=True,
                             dim_val=512,  
                             n_encoder_layers=4,
                             n_heads=8,
                             dropout_encoder=0.2, 
                             dropout_pos_enc=0.1,
                             dim_feedforward_encoder=2048,
                             num_predicted_features=1).to(device)

# optimizers
optimizer = torch.optim.Adam(encoder.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.5)

# train
encoder.train()
for i in range(500):
    print("Epoch:", i)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (features, gt_edges) in pbar:
        B = features.shape[0]
        optimizer.zero_grad()
        features = features.to(device)