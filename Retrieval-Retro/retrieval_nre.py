import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from models import GraphNetwork, GraphNetwork_prop
import utils_main
from collections import defaultdict
from tqdm import tqdm
import pickle
from collections import defaultdict


torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = "4"

def seed_everything(seed):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def make_retrieved(mode, split, rank_matrix, k, seed):
     

#     save_path = f'./output/new_retrieved_formation/sum_again_fast_0502_year_{mode}_seed_{seed}_retrieved_{k}'
    

#     candidate_list = defaultdict(list)

#     for idx, sim_row in enumerate(rank_matrix):
#         top_k_val, top_k_idx = torch.topk(sim_row, k, largest=False)
#         candidate_list[idx] = top_k_idx.tolist()

#     with open(save_path, 'w') as f:
#             json.dump(candidate_list, f)

def main():
    args = utils_main.parse_args()
    train_config = utils_main.training_config(args)
    configuration = utils_main.exp_get_name_RetroPLEX(train_config)
    print(f'configuration: {configuration}')

    K = args.K
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)
    seed_everything(seed=args.seed)

    nre_train = f'./dataset/{args.split}/train_nre_retrieved_{K}'
    nre_valid = f'./dataset/{args.split}/valid_nre_retrieved_{K}'
    nre_test = f'./dataset/{args.split}/test_nre_retrieved_{K}'

    with open(nre_train, 'r') as f:
        reaction_train = json.load(f)

    with open(nre_valid, 'r') as f:
        reaction_valid = json.load(f)

    with open(nre_test, 'r') as f:
        reaction_test = json.load(f)


    mpc_train = f'./dataset/{args.split}/train_mpc_retrieved_{K}'
    mpc_valid = f'./dataset/{args.split}/valid_mpc_retrieved_{K}'
    mpc_test = f'./dataset/{args.split}/test_mpc_retrieved_{K}'


    with open(mpc_train, 'r') as f:
        mpc_train = json.load(f)

    with open(mpc_valid, 'r') as f:
        mpc_valid = json.load(f)

    with open(mpc_test, 'r') as f:
        mpc_test = json.load(f)


    reaction_mpc_train = defaultdict(list)

    for idx, (reaction, mpc) in enumerate(zip(reaction_train, mpc_train.values())):
        
        if reaction is None:
            reaction = []
        else:
            reaction = reaction.copy()

        if len(reaction) < K:
            shortage = K - len(reaction)
            reaction.extend(mpc[:shortage])

        reaction_mpc_train[idx] = reaction[:K]
    
    save_path = f'./dataset/{args.split}/train_nre_final_retrieved_{K}'
    with open(save_path, 'w') as f:
        json.dump(reaction_mpc_train, f)

    reaction_mpc_valid = defaultdict(list)

    for idx, (reaction, mpc) in enumerate(zip(reaction_valid, mpc_valid.values())):
        
        if reaction is None:
            reaction = []
        else:
            reaction = reaction.copy()

        if len(reaction) < K:
            shortage = K - len(reaction)
            reaction.extend(mpc[:shortage])
        reaction_mpc_valid[idx] = reaction[:K]
    
    save_path = f'./dataset/{args.split}/valid_nre_final_retrieved_{K}'
    with open(save_path, 'w') as f:
        json.dump(reaction_mpc_valid, f)

    reaction_mpc_test = defaultdict(list)

    for idx, (reaction, mpc) in enumerate(zip(reaction_test, mpc_test.values())):
        
        if reaction is None:
            reaction = []
        else:
            reaction = reaction.copy()

        if len(reaction) < K:
            shortage = K - len(reaction)
            reaction.extend(mpc[:shortage])


        reaction_mpc_test[idx] = reaction[:K]

    save_path = f'./dataset/{args.split}/test_nre_final_retrieved_{K}'
    with open(save_path, 'w') as f:
        json.dump(reaction_mpc_test, f)



if __name__ == "__main__":

    main()