#!/usr/bin/env python3
# encoding utf-8
import argparse
import torch.multiprocessing as mp

from hfo import *
from Networks import ValueNetwork
from Worker import *
from SharedAdam import SharedAdam

from Environment import HFOEnv

# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment, and initialization of
# your models, torch's multiprocessing methods, etc.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--max_timestep', type=int, default=500)
    parser.add_argument('--iterate_target', type=int, default=500)
    parser.add_argument('--iterate_async', type=int, default=20)
    parser.add_argument('--discountFactor', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=1.)
    args = parser.parse_args()


    value_network = ValueNetwork(15,[15,15],4)
    saveModelNetwork(value_network, 'params/params_last')
    target_value_network = ValueNetwork(15,[15,15],4)
    optimizer = SharedAdam(value_network.parameters())

    # Example on how to initialize global locks for processes
    # and counters.
    counter = mp.Value('i', 500001)
    lock = mp.Lock()

    processes = []

# Example code to initialize torch multiprocessing.
    for idx in range(0, args.num_processes):
        trainingArgs = (idx, args, value_network, target_value_network, optimizer, lock, counter)
        p = mp.Process(target=train, args=trainingArgs)
        p.start()
        processes.append(p)

        for p in processes:
            p.join()

