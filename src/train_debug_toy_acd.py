import os
from itertools import chain
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F
import argparse
import pandas as pd
import json

from data_loaders import load_data
from models.MLPEncoder import MLPEncoder
from models.MLPDecoder import MLPDecoder
from utils.Pyutils import sample_gumbel, my_softmax, kl_categorical_uniform, encode_onehot, find_gpu_device, save_pkl

parser = argparse.ArgumentParser()
parser.add_argument("--simulation", type=str, default="springs", help="What simulation to generate.")
parser.add_argument("--num_atoms", type=int, default=5, help="Number of variables to consider..")
parser.add_argument("--temperature", type=float, default=0.1, help="Temperature of SpringSim simulation.")
parser.add_argument("--length", type=int, default=1000, help="Length of trajectory.")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of training simulations to generate.",)
parser.add_argument("--n_lags", type=int, default=None, help="Number of lags in the simulation (Econ only).")
parser.add_argument("--n_epochs", type=int, default=500, help="Number of epochs to use..")

if __name__ == "__main__":
    args = parser.parse_args()

    # running parameters
    model_name = "mlp_encoder_decoder"
    device = torch.device(find_gpu_device())
    num_atoms = args.num_atoms
    length = args.length
    n_lags = args.n_lags
    num_samples = args.num_samples
    dataset_name = args.simulation
    n_epochs = args.n_epochs
    learning_rate = 5e-4
    
    suffix = "_" + args.simulation + str(num_atoms)

    if (args.temperature is not None):
        suffix += "_inter" + str(args.temperature)

    if (args.length is not None):
        suffix += "_l" + str(args.length)

    if args.num_samples != 50000:
        suffix += "_s" + str(args.num_samples)

    if args.n_lags is not None:
        suffix += "_lag" + str(args.n_lags)
    
    root_path = os.path.dirname(__file__)
    data_path = os.path.join(root_path, "data")

    ## load data
    train_dataset = load_data(root_dir=os.path.join(data_path, dataset_name),
                              suffix=suffix,
                              num_atoms=num_atoms)
    
    # NOTE: Random sampling occurs in the "num_sample" dimension instead of "num_obs"
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    off_diag = np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)
    # rel_rec, rel_send: [num_atoms * (num_atoms - 1), num_atoms]
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
    rel_rec = torch.FloatTensor(rel_rec).to(device)
    rel_send = torch.FloatTensor(rel_send).to(device)

    # load Models
    encoder = MLPEncoder(train_dataset.tensors[0].shape[-2] * train_dataset.tensors[0].shape[-1], 256, 2).to(device)
    decoder = MLPDecoder(4, 256, 2).to(device)

    # optimizers
    optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.5)
    # train
    train_acc_vals = []

    encoder.train()
    decoder.train()
    for i in range(n_epochs):
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

            train_acc_vals.append(edge_acc)

            pbar.set_description("Loss: {:.4e}, Edge Acc: {:2f}, MSE: {:4e}".format(loss.item(), edge_acc, F.mse_loss(output, decode_target).item()))
        scheduler.step()

    acc_df = pd.DataFrame(train_acc_vals, columns=["acc"])

    results = {
        
        "acc": acc_df,

        }

    output_path = os.path.join(os.path.dirname(__file__),
                               "results",
                               model_name)

    # check if dir exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save args
    args_dict = vars(args)  
    with open(os.path.join(output_path, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)

    # save results
    save_pkl(data=results, path=os.path.join(output_path, "results.pickle"))
            

