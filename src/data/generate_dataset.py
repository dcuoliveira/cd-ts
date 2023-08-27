"""Based on https://github.com/ethanfetaya/NRI (MIT License)."""
import sys
import os
import json
import time
import numpy as np
import argparse
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from data.synthetic_sim import SpringSim
from data.economic_sim import EconomicSim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_only", type=bool, default=True, help="If to generate train data only."
    )
    parser.add_argument(
        "--simulation", type=str, default="econ", help="What simulation to generate."
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=100,
        help="Number of training simulations to generate.",
    )
    parser.add_argument(
        "--num_valid",
        type=int,
        default=10000,
        help="Number of validation simulations to generate.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=10000,
        help="Number of test simulations to generate.",
    )
    parser.add_argument(
        "--length", type=int, default=1000, help="Length of trajectory."
    )
    parser.add_argument(
        "--sample_freq",
        type=int,
        default=100,
        help="How often to sample the trajectory.",
    )
    parser.add_argument(
        "--n_balls", type=int, default=5, help="Number of balls in the simulation."
    )
    parser.add_argument(
        "--n_lags", type=int, default=2, help="Number of lags in the simulation (Econ only)."
    )
    parser.add_argument(
        "--sparsity_threshold", type=float, default=1-0.95, help="Percentual of zeros in the adj matrix (Econ only)."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.") # 42
    parser.add_argument(
        "--datadir", type=str, default=os.path.dirname(__file__), help="Name of directory to save data to."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None, # 0.1
        help="Temperature of SpringSim simulation.",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Have symmetric connections (non-causal)",
    )
    parser.add_argument(
        "--fixed_particle",
        action="store_true",
        default=False,
        help="Have one particle fixed in place and influence all others",
    )
    parser.add_argument(
        "--influencer_particle",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) influences all",
    )
    parser.add_argument(
        "--confounder",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) influences at least two others",
    )
    parser.add_argument(
        "--uninfluenced_particle",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) is not influence by others",
    )
    parser.add_argument(
        "--fixed_connectivity",
        action="store_true",
        default=False,
        help="Have one inherent causal structure for ALL simulations",
    )
    args = parser.parse_args()

    if args.fixed_particle:
        args.influencer_particle = True
        args.uninfluenced_particle = True

    assert not (args.confounder and args.influencer_particle), "These options are mutually exclusive."

    args.length_test = args.length * 2

    print(args)
    return args


def generate_dataset(num_sims, length, sample_freq, sampled_sims=None):
    if not sampled_sims is None:
        assert len(sampled_sims) == num_sims

    loc_all = list()
    vel_all = list()
    edges_all = list()

    if args.fixed_connectivity:
        edges = sim.get_edges(
            undirected=args.undirected,
            influencer=args.influencer_particle,
            uninfluenced=args.uninfluenced_particle,
            confounder=args.confounder
        )
        print('\x1b[5;30;41m' + "Edges are fixed to be: " + '\x1b[0m')
        print(edges)
    else:
        edges = None

    for i in range(num_sims):
        if not sampled_sims is None:
            sim_i = sampled_sims[i]
        else:
            sim_i = sim
        t = time.time()
        loc, vel, edges = sim_i.sample_trajectory(
            T=length,
            sample_freq=sample_freq,
            undirected=args.undirected,
            fixed_particle=args.fixed_particle,
            influencer=args.influencer_particle,
            uninfluenced=args.uninfluenced_particle,
            confounder=args.confounder,
            edges=edges
        )
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

        if not args.fixed_connectivity:
            edges = None

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all

def generate_econ_dataset(num_samples, num_balls, T, num_lags):

    # feats: (num_samples, num_objects, num_timesteps, num_feature_per_obj)
    feats = torch.zeros((num_samples, num_balls, T, 1))
    # edges_all: (num_samples, num_objects, num_objects * num_lags)
    edges_all = torch.zeros((num_samples, num_balls, num_balls * num_lags))
    # phi_all: (num_samples, num_objects, num_objects * num_lags)
    phi_all = torch.zeros((num_samples, num_balls, num_balls * num_lags))

    pbar = tqdm(range(num_samples), total=num_samples)
    for s in pbar:

        # simulate VAR process
        t = time.time()
        var_sim, phi = sim.simulate_VAR(seed=s)

        # add time series to each dimension
        for i in range(num_balls):
            feats[s, i, :, :] = torch.from_numpy(var_sim[:, i][:, None])

        # add phis
        phi_all[s, :, :] = torch.from_numpy(phi)

        # add phi into adjacency matrix
        # if phi[i, j] != 0, then there is an edge from j to i
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                if phi[i, j] != 0:
                    edges_all[s, i, j] = 1

        print("Iter: {}, Simulation time: {}".format(s, time.time() - t))

    return feats, edges_all, phi_all


if __name__ == "__main__":

    args = parse_args()

    if args.simulation == "springs":
        sim = SpringSim(
            noise_var=0.0,
            n_balls=args.n_balls,
            interaction_strength=args.temperature,
        )
    elif args.simulation == "econ":
        sim = EconomicSim(
            num_timesteps = args.length,
            num_objects = args.n_balls,
            num_lags = args.n_lags,
            sparsity_threshold = args.sparsity_threshold,
        )
    else:
        raise ValueError("Simulation {} not implemented".format(args.simulation))

    suffix = "_" + args.simulation

    suffix += str(args.n_balls)

    if args.undirected:
        suffix += "_undir"

    if args.fixed_particle:
        suffix += "_fixed"

    if args.uninfluenced_particle:
        suffix += "_uninfluenced"

    if args.influencer_particle:
        suffix += "_influencer"

    if args.confounder:
        suffix += "_conf"

    if (args.temperature is not None):
        suffix += "_inter" + str(args.temperature)

    if (args.length is not None):
        suffix += "_l" + str(args.length)

    if (args.num_train is not None):
        suffix += "_s" + str(args.num_train)

    if args.fixed_connectivity:
        suffix += "_oneconnect"

    if args.n_lags is not None:
        suffix += "_lag" + str(args.n_lags)

    print(suffix)

    # check if data path exists
    if not os.path.exists(args.datadir):
        os.makedirs(args.datadir)

    # check if simulation dir exists
    if not os.path.exists(os.path.join(args.datadir, args.simulation)):
        os.makedirs(os.path.join(args.datadir, args.simulation))

    args.datadir = os.path.join(args.datadir, args.simulation)
    json.dump(
        vars(args),
        open(os.path.join(args.datadir, "args.json"), "w"),
        indent=4,
        separators=(",", ": "),
    )

    if args.train_only:
        print("Generating {} training simulations".format(args.num_train))

        if args.simulation == "econ":
            loc_train, edges_train, phis_train = generate_econ_dataset(
                num_balls=args.n_balls,
                num_samples=args.num_train,
                T=args.length,
                num_lags=args.n_lags,
            )
            vel_train = None
        else:
            np.random.seed(args.seed)
            loc_train, vel_train, edges_train = generate_dataset(
                args.num_train,
                args.length,
                args.sample_freq,
                sampled_sims=(None),
            )
            phis_train = None

        np.save(os.path.join(args.datadir, "loc_train" + suffix + ".npy"), loc_train)
        np.save(os.path.join(args.datadir, "vel_train" + suffix + ".npy"), vel_train)
        np.save(os.path.join(args.datadir, "edges_train" + suffix + ".npy"), edges_train)
        np.save(os.path.join(args.datadir, "phis_train" + suffix + ".npy"), phis_train)

    else:
        raise NotImplementedError("Not implemented")

        print("Generating {} training simulations".format(args.num_train))
        loc_train, vel_train, edges_train = generate_dataset(
            args.num_train,
            args.length,
            args.sample_freq,
            sampled_sims=(None),
        )
        np.save(os.path.join(args.datadir, "loc_train" + suffix + ".npy"), loc_train)
        np.save(os.path.join(args.datadir, "vel_train" + suffix + ".npy"), vel_train)
        np.save(os.path.join(args.datadir, "edges_train" + suffix + ".npy"), edges_train)
        
        print("Generating {} validation simulations".format(args.num_valid))
        loc_valid, vel_valid, edges_valid = generate_dataset(
            args.num_valid,
            args.length,
            args.sample_freq,
            sampled_sims=(None),
        )
        np.save(os.path.join(args.datadir, "loc_valid" + suffix + ".npy"), loc_valid)
        np.save(os.path.join(args.datadir, "vel_valid" + suffix + ".npy"), vel_valid)
        np.save(os.path.join(args.datadir, "edges_valid" + suffix + ".npy"), edges_valid)

        print("Generating {} test simulations".format(args.num_test))
        loc_test, vel_test, edges_test = generate_dataset(
            args.num_test,
            args.length_test,
            args.sample_freq,
            sampled_sims=(None),
        )
        np.save(os.path.join(args.datadir, "loc_test" + suffix + ".npy"), loc_test)
        np.save(os.path.join(args.datadir, "vel_test" + suffix + ".npy"), vel_test)
        np.save(os.path.join(args.datadir, "edges_test" + suffix + ".npy"), edges_test)
