from itertools import chain
import numpy as np
import torch
from tqdm import tqdm

from data_loaders import load_springs_data
from models.GNNs import NRIMLP
from models.MLPDecoder import MLPDecoder
from utils.Pyutils import my_softmax, kl_categorical_uniform, edge_accuracy, encode_onehot, expand_edges, get_off_diag_idx

def sample_gumbel(logits, temperature=1.0):
    gumbel_noise = -torch.log(1e-10 - torch.log(torch.rand_like(logits) + 1e-10))
    y = logits + gumbel_noise
    return torch.softmax(y / temperature, axis=-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Load data
train_dataset = load_springs_data("src/data/spring_data_test", "_springs3_l5100_s1000", num_atoms=3)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
off_diag = np.ones((3, 3)) - np.eye(3)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec).to(device)
rel_send = torch.FloatTensor(rel_send).to(device)
## Load Models
encoder = NRIMLP(50*4, 128, 2)
decoder = MLPDecoder(4, 128, 2)
## Optimizers
optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=7e-4)
## Train
encoder.train()
decoder.train()
for i in range(100):
    print("Epoch:", i)
    pbar = tqdm((train_loader), total=len(train_loader))
    for i, (features, edges) in pbar:
        
        optimizer.zero_grad()
        features = features.to(device)
        edges = edges.to(device)
        logits = encoder.forward(features, rel_rec=rel_rec, rel_send=rel_send)
        prob = my_softmax(logits, -1)
        
                # TODO
        # Gumbel-Softmax sampling
        edges = sample_gumbel(logits, temperature=0.5)
        # Decoding step
        output = decoder(features, edges, rel_rec=rel_rec, rel_send=rel_send)
        decode_target = features[:,:,1:]

        loss = kl_categorical_uniform(preds=prob,
                                        num_atoms=features.shape[1],
                                        num_edge_types=2)
        #loss += nll_likelihood(output, decode_target, var=5e-4)
        loss.backward()
        optimizer.step()

