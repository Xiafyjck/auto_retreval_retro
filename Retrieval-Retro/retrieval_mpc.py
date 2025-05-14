import torch
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.loader import DataLoader
import json
import torch.nn.functional as F
from collections import defaultdict
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='year')
parser.add_argument('--K', type=int, default=3, help='number of candidates to retrieve')
parser.add_argument('--device', type=int, default=0, help='GPU device ID')
args = parser.parse_args()

def make_sim_mpc(device):

    # Load saved embeddings from trained MPC (Fisrt, you need to train the MPC ,then save the embeddings)
    load_path_train = f'./dataset/{args.split}/train_mpc_embeddings.pt'
    load_path_valid = f'./dataset/{args.split}/valid_mpc_embeddings.pt'
    load_path_test = f'./dataset/{args.split}/test_mpc_embeddings.pt'

    train_emb = torch.load(load_path_train, map_location = device, weights_only=False).squeeze(1)
    valid_emb = torch.load(load_path_valid, map_location = device, weights_only=False).squeeze(1)
    test_emb = torch.load(load_path_test, map_location = device, weights_only=False).squeeze(1)

    train_emb_norm = F.normalize(train_emb, p=2, dim=1)
    valid_emb_norm = F.normalize(valid_emb, p=2, dim=1)
    test_emb_norm = F.normalize(test_emb, p=2, dim=1)

    cos_sim_train = torch.mm(train_emb_norm, train_emb_norm.t())
    cos_sim_valid = torch.mm(valid_emb_norm, train_emb_norm.t())
    cos_sim_test = torch.mm(test_emb_norm, train_emb_norm.t())

    diag_mask = torch.ones_like(cos_sim_train).to(device) - torch.eye(cos_sim_train.size(0), dtype=torch.float32).to(device)
    cos_sim_train= cos_sim_train * diag_mask

    torch.save(cos_sim_train, f"./dataset/{args.split}/train_mpc_cos_sim_matrix.pt")
    torch.save(cos_sim_valid, f"./dataset/{args.split}/valid_mpc_cos_sim_matrix.pt")
    torch.save(cos_sim_test, f"./dataset/{args.split}/test_mpc_cos_sim_matrix.pt")

    print(f'cosine similarity matrix mpc saving completed')


def compute_rank_in_batches(tensor, batch_size):

    # Zeros for the ranked tensor
    ranked_tensor = torch.zeros_like(tensor)
    
    # Compute the number of batches
    num_batches = tensor.size(0) // batch_size + (1 if tensor.size(0) % batch_size else 0)
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, tensor.size(0))
        batch = tensor[batch_start:batch_end]
        # Perform the ranking operation on the smaller batch
        batch_ranked = batch.argsort(dim=1).argsort(dim=1)
        ranked_tensor[batch_start:batch_end] = batch_ranked
    
    return ranked_tensor

def make_retrieved(mode, rank_matrix, k):
     
    save_path = f'./dataset/{args.split}/{mode}_mpc_retrieved_{k}'
    
    print(f"保存检索结果到: {save_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    candidate_list = defaultdict(list)

    for idx, sim_row in enumerate(rank_matrix):
        top_k_val, top_k_idx = torch.topk(sim_row, k)
        candidate_list[idx] = top_k_idx.tolist()

    with open(save_path, 'w') as f:
        json.dump(candidate_list, f)
    
    print(f"已保存 {len(candidate_list)} 个样本的检索结果到 {save_path}")



def main():

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(f"使用设备: {device}")
    print(f"数据集拆分: {args.split}")
    print(f"检索数量K: {args.K}")

    make_sim_mpc(device)

    yr_mpc_train = torch.load(f"./dataset/{args.split}/train_mpc_cos_sim_matrix.pt", map_location=device, weights_only=False)
    yr_mpc_valid = torch.load(f"./dataset/{args.split}/valid_mpc_cos_sim_matrix.pt", map_location=device, weights_only=False)
    yr_mpc_test = torch.load(f"./dataset/{args.split}/test_mpc_cos_sim_matrix.pt", map_location=device, weights_only=False)

    print(f"余弦相似度矩阵形状: train={yr_mpc_train.shape}, valid={yr_mpc_valid.shape}, test={yr_mpc_test.shape}")

    batch_size = 1000
    print(f"开始计算排名，批次大小：{batch_size}")
    
    rank_mpc_train = compute_rank_in_batches(yr_mpc_train, batch_size)
    print(f"训练集排名计算完成，形状: {rank_mpc_train.shape}")
    
    rank_mpc_valid = compute_rank_in_batches(yr_mpc_valid, batch_size)
    print(f"验证集排名计算完成，形状: {rank_mpc_valid.shape}")
    
    rank_mpc_test = compute_rank_in_batches(yr_mpc_test, batch_size)
    print(f"测试集排名计算完成，形状: {rank_mpc_test.shape}")

    rank_matrix_list = [rank_mpc_train, rank_mpc_valid, rank_mpc_test]


    for idx, matrix in enumerate(rank_matrix_list):

        if idx == 0:
            mode = 'train'
        elif idx == 1:
            mode = 'valid'
        elif idx == 2:
            mode = 'test'

        print(f"为{mode}集生成检索结果，K={args.K}")
        make_retrieved(mode, matrix, args.K)


if __name__ == "__main__":

    main()