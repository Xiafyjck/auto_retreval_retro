import torch
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.loader import DataLoader
import json
import argparse
from tqdm import tqdm
import os

# 设置命令行参数
parser = argparse.ArgumentParser(description='整合MPC和NRE检索结果')
parser.add_argument('--split', type=str, default='year', help='数据集拆分名称')
parser.add_argument('--mode', type=str, choices=['train', 'valid', 'test'], help='处理模式: train/valid/test')
parser.add_argument('--K', type=int, default=3, help='检索的候选数量')
parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
args = parser.parse_args()

def main():
    # 使用传入的参数
    mode = args.mode  
    K = args.K
    split = args.split
    
    # 设置设备
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(f"使用设备: {device}")
    
    print(f"数据集拆分: {split}")
    print(f"处理模式: {mode}")
    print(f"检索数量K: {K}")

    # 构建文件路径
    dataset_dir = f"./dataset/{split}"
    train_mpc_path = f"{dataset_dir}/train_mpc.pt"
    valid_mpc_path = f"{dataset_dir}/valid_mpc.pt"
    test_mpc_path = f"{dataset_dir}/test_mpc.pt"
    
    # 检查输入文件是否存在
    mpc_retrieval_path = f"{dataset_dir}/{mode}_mpc_retrieved_{K}"
    nre_retrieval_path = f"{dataset_dir}/{mode}_nre_retrieved_{K}"
    
    if not os.path.exists(mpc_retrieval_path):
        print(f"错误: MPC检索结果不存在: {mpc_retrieval_path}")
        return
        
    if not os.path.exists(nre_retrieval_path):
        print(f"错误: NRE检索结果不存在: {nre_retrieval_path}")
        return
    
    print(f"加载数据集...")
    # 加载数据集
    try:
        train_dataset = torch.load(train_mpc_path, weights_only=False)
        valid_dataset = torch.load(valid_mpc_path, weights_only=False)
        test_dataset = torch.load(test_mpc_path, weights_only=False)
        
        print(f"加载MPC检索结果: {mpc_retrieval_path}")
        with open(mpc_retrieval_path, "r") as f:
            candi_data = json.load(f)

        print(f"加载NRE检索结果: {nre_retrieval_path}")
        with open(nre_retrieval_path, "r") as f:
            candi_data_2 = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载数据: {e}")
        return
    
    # 选择当前处理的数据集
    if mode == "train":
        dataset = train_dataset
    elif mode == "valid":
        dataset = valid_dataset
    elif mode == "test":
        dataset = test_dataset
    
    print(f"开始整合检索结果，数据集大小: {len(dataset)}")
    new_data = []

    for idx, data in enumerate(tqdm(dataset, desc=f"处理{mode}数据集")):
        tmp = [data]
        
        # 确保索引存在于检索结果中
        idx_str = str(idx)
        if idx_str not in candi_data or idx_str not in candi_data_2:
            print(f"警告: 索引 {idx} 在检索结果中不存在")
            continue
            
        candi_idx = candi_data[idx_str]
        candi_idx_2 = candi_data_2[idx_str]
        
        # 创建子图容器
        subgraph = []
        subgraph_2 = []
        
        # 处理MPC检索结果
        for i in candi_idx:
            if i >= len(train_dataset):
                print(f"警告: MPC候选索引 {i} 超出训练集范围")
                continue
                
            try:
                x = train_dataset[i].x
                edge_index = train_dataset[i].edge_index
                fc_weight = train_dataset[i].fc_weight
                edge_attr = train_dataset[i].edge_attr
                comp_fea = train_dataset[i].comp_fea
                y = train_dataset[i].y_lb_one
                
                candi_graph = Data(
                    x=x, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    fc_weight=fc_weight, 
                    comp_fea=comp_fea, 
                    precursor=y
                )
                subgraph.append(candi_graph)
            except Exception as e:
                print(f"处理MPC候选 {i} 时出错: {e}")
        
        # 处理NRE检索结果
        for i in candi_idx_2:
            if i >= len(train_dataset):
                print(f"警告: NRE候选索引 {i} 超出训练集范围")
                continue
                
            try:
                x = train_dataset[i].x
                edge_index = train_dataset[i].edge_index
                fc_weight = train_dataset[i].fc_weight
                edge_attr = train_dataset[i].edge_attr
                comp_fea = train_dataset[i].comp_fea
                y = train_dataset[i].y_lb_one
                
                candi_graph2 = Data(
                    x=x, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    fc_weight=fc_weight, 
                    comp_fea=comp_fea, 
                    precursor=y
                )
                subgraph_2.append(candi_graph2)
            except Exception as e:
                print(f"处理NRE候选 {i} 时出错: {e}")

        # 将所有结果添加到数据中
        tmp.append(subgraph)
        tmp.append(subgraph_2)
        new_data.append(tuple(tmp))

    # 保存整合结果
    output_path = f"{dataset_dir}/{mode}_final_mpc_nre_K_{K}.pt"
    print(f"保存整合结果到: {output_path}, 共 {len(new_data)} 个样本")
    torch.save(new_data, output_path)
    print(f"{mode} 数据集整合完成!")

if __name__ == "__main__":
    main()
                
