import argparse
import random
import numpy as np
import pandas as pd

# this code generates the specified synthetic dataset

# Subset in R^D of dimension d with gaussian noise of sigma (?and some seed) and size N

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--dataset', default='HV', type=str, choices=['HV', 'HP', 'BHP', 'TC'])
    parser.add_argument('--N', default=2500, type=int)
    parser.add_argument('--D', default=100, type=int)
    parser.add_argument('--d', default=10, type=int)
    
    args, _ = parser.parse_known_args()
    return args

def biased_hyperplane(args):

    clean_signal = []

    for i in range(int(args.N/2)):
        
        weights = np.random.uniform(-1, 1, args.d) # assign weights following some distribution
        sample = weights.tolist() # linear combination of unitary vectors
        sample += [0]*(args.D-args.d)

        clean_signal += [sample]

    for i in range(int(args.N/2)):

        weights = np.random.uniform(0, 1, args.d)   # more points on the positive side
        sample = weights.tolist() # linear combination of unitary vectors
        sample += [0]*(args.D-args.d)

        clean_signal += [sample]

    return clean_signal

def two_clusters(args):

    clean_signal = []

    for i in range(int(args.N/4)):
        
        weights = np.random.uniform(-2, -1, args.d) # assign weights following some distribution
        sample = weights.tolist() # linear combination of unitary vectors
        sample += [0]*(args.D-args.d)

        clean_signal += [sample]

    for i in range(int(3*args.N/4)):

        weights = np.random.uniform(1, 2, args.d)   # more points on the positive side
        sample = weights.tolist() # linear combination of unitary vectors
        sample += [0]*(args.D-args.d)

        clean_signal += [sample]

    return clean_signal

def hyperplane(args):

    clean_signal = []

    for i in range(args.N):
        
        weights = np.random.uniform(-1, 1, args.d) # assign weights following some distribution
        sample = weights.tolist() # linear combination of unitary vectors
        sample += [0]*(args.D-args.d)

        clean_signal += [sample]

    return clean_signal

def hypervolume(args):

    clean_signal = []

    for i in range(args.N):
        
        weights = np.random.uniform(-1, 1, args.D) # assign weights following some distribution
        sample = weights.tolist() # linear combination of unitary vectors

        clean_signal += [sample]

    return clean_signal

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == 'BHP':
        clean_signal = biased_hyperplane(args)
    elif args.dataset == 'HP':
        clean_signal = hyperplane(args)
    elif args.dataset == 'HV':
        clean_signal = hypervolume(args)
    elif args.dataset == 'TC':
        clean_signal = two_clusters(args)

    

    # sigma = 0.01
    # noise = np.random.normal(0, sigma, [args.N, args.D]) 
    signal = clean_signal #+ noise

    df = pd.DataFrame(signal)
    df = df.transpose()
    df.to_csv(f"data/synthetic/{args.dataset}_{args.N}_{args.D}_{args.d}.csv", index=False, header=False)
