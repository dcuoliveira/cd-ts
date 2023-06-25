import os
from itertools import chain
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F

from data_loaders import load_springs_data
from models.MLPEncoder import MLPEncoder
from models.MLPDecoder import MLPDecoder
from utils.Pyutils import my_softmax, kl_categorical_uniform, encode_onehot, find_gpu_device

def sample_gumbel(logits, tau=1.0):
    gumbel_noise = -torch.log(1e-10 - torch.log(torch.rand_like(logits) + 1e-10))
    y = logits + gumbel_noise
    return torch.softmax(y / tau, axis=-1)

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
encoder = MLPEncoder(49*4, 256, 2).to(device)
decoder = MLPDecoder(4, 256, 2).to(device)

# optimizers
optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.5)
# train
encoder.train()
decoder.train()
for i in range(500):
    print("Epoch:", i)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (features, gt_edges) in pbar:
        B = features.shape[0]
        optimizer.zero_grad()
        features = features.to(device)
        gt_edges = gt_edges.to(device)

        # train encoder
        ## features: (batch_size, num_objects, num_timesteps, num_feature_per_obj)
        logits = encoder.forward(features, rel_rec=rel_rec, rel_send=rel_send)
        # NOTE - Why do we have 2*num_objects instead of num_objects?
        ## logits: (batch_size, 2*num_objects)
        prob = my_softmax(logits, -1)
        
        # gumbel-softmax sampling
        edges = sample_gumbel(logits, tau=0.5)

        # decoding step
        output = decoder.forward(features, edges, rel_rec=rel_rec, rel_send=rel_send, teacher_forcing=10)
        decode_target = features[:, :, 1:]

        loss = kl_categorical_uniform(preds=prob,
                                      num_atoms=num_atoms,
                                      num_edge_types=2)
        distrib = torch.distributions.Normal(output, 5e-7)
        loss -= distrib.log_prob(decode_target).sum()/B
        loss.backward()
        optimizer.step()
        edge_acc = torch.sum(edges.argmax(-1) == gt_edges).item()/(B*num_atoms*(num_atoms-1))
        pbar.set_description("Loss: {:.4e}, Edge Acc: {:2f}, MSE: {:4e}".format(loss.item(), edge_acc, F.mse_loss(output, decode_target).item()))
    scheduler.step()
        

