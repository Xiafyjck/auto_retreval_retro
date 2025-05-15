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
import time

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

def make_retrieved(mode, split, rank_matrix, k, seed):
    # 修正保存路径，使其与RetrievalRetroPipeline.sh预期的路径一致
    save_path = f'./dataset/{split}/{mode}_nre_retrieved_{k}'
    
    print(f"保存NRE检索结果到: {save_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    candidate_list = defaultdict(list)
    
    # 检查数据大小
    memory_required = rank_matrix.shape[0] * rank_matrix.shape[1] * 4 / (1024**3)  # 估计内存需求（GB）
    print(f"矩阵大小: {rank_matrix.shape}, 估计内存需求: {memory_required:.2f} GB")
    
    # 如果矩阵太大，使用分批处理
    if memory_required > 1.0:  # 如果需要超过1GB内存
        print(f"矩阵较大，使用分批处理...")
        batch_size = max(1, min(1000, rank_matrix.shape[0] // 10))  # 自适应批大小
        
        for i in range(0, rank_matrix.shape[0], batch_size):
            end_idx = min(i + batch_size, rank_matrix.shape[0])
            print(f"处理批次 {i}-{end_idx} / {rank_matrix.shape[0]}")
            
            # 将当前批次移到设备上
            batch_matrix = rank_matrix[i:end_idx].to(rank_matrix.device)
            
            for idx, sim_row in enumerate(tqdm(batch_matrix, desc=f"批次 {i}-{end_idx} 检索")):
                top_k_val, top_k_idx = torch.topk(sim_row, k, largest=False)
                candidate_list[i + idx] = top_k_idx.tolist()
    else:
        # 如果矩阵不大，一次性处理
        for idx, sim_row in enumerate(tqdm(rank_matrix, desc=f"检索 {mode} 集")):
            top_k_val, top_k_idx = torch.topk(sim_row, k, largest=False)
            candidate_list[idx] = top_k_idx.tolist()

    with open(save_path, 'w') as f:
        json.dump(candidate_list, f)
    
    print(f"已保存 {len(candidate_list)} 个样本的NRE检索结果到 {save_path}")

def main():
    args = utils_main.parse_args()
    train_config = utils_main.training_config(args)
    configuration = utils_main.exp_get_name_RetroPLEX(train_config)
    print(f'configuration: {configuration}')

    # 添加更详细的目录和文件检查
    print(f"\n=== 路径检查 ===")
    dataset_dir = f"./dataset/{args.split}"
    print(f"数据集目录: {dataset_dir}")
    
    # 检查数据集目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"警告: 数据集目录 {dataset_dir} 不存在! 尝试创建...")
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"已创建数据集目录: {dataset_dir}")
        except Exception as e:
            print(f"错误: 无法创建数据集目录: {e}")
    
    # 检查必要的数据文件
    required_files = [
        f"./dataset/{args.split}/precursor_graph.pt",
        f"./dataset/{args.split}/train_mpc.pt",
        f"./dataset/{args.split}/valid_mpc.pt",
        f"./dataset/{args.split}/test_mpc.pt"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"✗ 文件不存在: {file_path}")
            all_files_exist = False
    
    if not all_files_exist:
        print("警告: 一些必要的文件不存在，这可能导致脚本执行失败。")
        print("请确保已经运行了前面的数据准备步骤，并且文件已经正确复制到相应目录。")
        
    # 检查预期的输出路径
    output_files = [
        f"./dataset/{args.split}/train_nre_retrieved_{args.K}",
        f"./dataset/{args.split}/valid_nre_retrieved_{args.K}",
        f"./dataset/{args.split}/test_nre_retrieved_{args.K}"
    ]
    
    print("\n预期的输出文件:")
    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"✓ 已存在: {file_path}")
        else:
            print(f"- 将创建: {file_path}")
    
    # 使用命令行参数中的设备ID，不再硬编码
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(f"\n使用设备: {device}")
    print(f"数据集拆分: {args.split}")
    print(f"检索数量K: {args.K}")
    
    seed_everything(seed=args.seed)

    print("\n加载数据集...")
    start_time = time.time()
    
    try:
        precursor_graph = torch.load(f"./dataset/{args.split}/precursor_graph.pt", map_location=device, weights_only=False)
        precursor_loader = DataLoader(precursor_graph, batch_size = 1, shuffle=False)
        
        train_dataset = torch.load(f'./dataset/{args.split}/train_mpc.pt', weights_only=False)
        valid_dataset = torch.load(f'./dataset/{args.split}/valid_mpc.pt', weights_only=False)
        test_dataset = torch.load(f'./dataset/{args.split}/test_mpc.pt', weights_only=False)
        
        train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size = 1)
        test_loader = DataLoader(test_dataset, batch_size = 1)
        
        print(f"数据集加载完成! 用时: {time.time() - start_time:.2f}秒")
        print(f"数据集大小: 前驱体={len(precursor_graph)}, 训练集={len(train_dataset)}, 验证集={len(valid_dataset)}, 测试集={len(test_dataset)}")
    except Exception as e:
        print(f"错误: 无法加载数据集: {e}")
        print("请确保数据文件存在且格式正确。")
        return
        
    try:
        n_hidden = args.hidden
        n_atom_feat = train_dataset[0].x.shape[1]
        n_bond_feat = train_dataset[0].edge_attr.shape[1]
        output_dim = train_dataset[0].y_multiple.shape[1] #Dataset precursor set dim 
    except Exception as e:
        print(f"错误: 无法获取数据特征维度: {e}")
        return

    print("加载预训练模型...")
    model = GraphNetwork_prop(args.layers, n_atom_feat, n_bond_feat, n_hidden, device).to(device)
    checkpoint = torch.load("./dataset/TL_pretrain(formation_exp)_embedder(graphnetwork)_lr(0.0005)_batch_size(256)_hidden(256)_seed(0)_.pt", map_location = device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f'模型加载完成!')

    ### Calculating formation energy for precursor graph ###
    print("\n计算形成能...")
    formation_start_time = time.time()
    
    # 检查是否已存在形成能文件，如果存在则跳过计算
    all_formation_files_exist = all([
        os.path.exists(f'./dataset/{args.split}/precursor_formation_energy.pt'),
        os.path.exists(f'./dataset/{args.split}/train_formation_energy.pt'),
        os.path.exists(f'./dataset/{args.split}/valid_formation_energy.pt'),
        os.path.exists(f'./dataset/{args.split}/test_formation_energy.pt')
    ])
    
    if all_formation_files_exist:
        print("所有形成能文件已存在，跳过形成能计算步骤")
    else:
        precursor_formation_list = []
        train_formation_list = []
        valid_formation_list = []
        test_formation_list = []

        model.eval()
        with torch.no_grad():
            print("计算前驱体形成能...")
            for bc, batch in enumerate(tqdm(precursor_loader, desc="前驱体形成能")):
                batch.to(device)
                y,_ = model(batch)
                precursor_formation_list.append(y)
            precursor_y_tensor = torch.stack(precursor_formation_list)
            torch.save(precursor_y_tensor, f'./dataset/{args.split}/precursor_formation_energy.pt')
            print(f"前驱体形成能已保存，形状: {precursor_y_tensor.shape}")

            print("计算训练集形成能...")
            for bc, batch in enumerate(tqdm(train_loader, desc="训练集形成能")):
                batch.to(device)
                y,_ = model(batch)
                train_formation_list.append(y)
            train_y_tensor = torch.stack(train_formation_list)
            torch.save(train_y_tensor, f'./dataset/{args.split}/train_formation_energy.pt')
            print(f"训练集形成能已保存，形状: {train_y_tensor.shape}")

            print("计算验证集形成能...")
            for bc, batch in enumerate(tqdm(valid_loader, desc="验证集形成能")):
                batch.to(device)
                y,_ = model(batch)
                valid_formation_list.append(y)
            valid_y_tensor = torch.stack(valid_formation_list)
            torch.save(valid_y_tensor, f'./dataset/{args.split}/valid_formation_energy.pt')
            print(f"验证集形成能已保存，形状: {valid_y_tensor.shape}")

            print("计算测试集形成能...")
            for bc, batch in enumerate(tqdm(test_loader, desc="测试集形成能")):
                batch.to(device)
                y,_ = model(batch)
                test_formation_list.append(y)
            test_y_tensor = torch.stack(test_formation_list)
            torch.save(test_y_tensor, f'./dataset/{args.split}/test_formation_energy.pt')
            print(f"测试集形成能已保存，形状: {test_y_tensor.shape}")
        
        print(f"形成能计算完成! 总用时: {time.time() - formation_start_time:.2f}秒")
    
    print("\n加载形成能数据...")
    precursor_formation_y = torch.load(f'./dataset/{args.split}/precursor_formation_energy.pt',map_location=device, weights_only=False)
    train_formation_y = torch.load(f'./dataset/{args.split}/train_formation_energy.pt', map_location=device, weights_only=False)
    valid_formation_y = torch.load(f'./dataset/{args.split}/valid_formation_energy.pt', map_location=device, weights_only=False)
    test_formation_y = torch.load(f'./dataset/{args.split}/test_formation_energy.pt', map_location=device, weights_only=False)
    K = args.K
    
    print(f"形成能数据加载完成! 形状: 前驱体={precursor_formation_y.shape}, 训练集={train_formation_y.shape}, 验证集={valid_formation_y.shape}, 测试集={test_formation_y.shape}")

    # 预先计算每个训练集样本的前驱体能量总和，避免重复计算
    print("\n预计算训练集样本的前驱体能量总和...")
    precursor_sums = []
    for db in tqdm(train_dataset, desc="计算前驱体能量总和"):
        precursor_indices = db.y_lb_one.nonzero(as_tuple=False).squeeze()
        if precursor_indices.dim() == 0:  # 处理只有一个前驱体的情况
            precursor_indices = precursor_indices.unsqueeze(0)
        precursor_energies = precursor_formation_y[precursor_indices]
        precursor_sum = precursor_energies.sum()
        precursor_sums.append(precursor_sum)
    precursor_sums = torch.tensor(precursor_sums, device=device)
    print("前驱体能量总和计算完成!")

    # For Train
    print("\n计算训练集形成能差异...")
    diff_start_time = time.time()
    
    # 检查是否已存在差异文件
    if os.path.exists(f'./dataset/{args.split}/train_formation_energy_calculation_delta_G.pt'):
        print(f"训练集差异文件已存在，跳过计算")
        # 分批加载以节省内存
        train_matrix = torch.load(f'./dataset/{args.split}/train_formation_energy_calculation_delta_G.pt', map_location='cpu', weights_only=False)
        # 只将用于处理的部分移到GPU
        train_matrix = train_matrix.to(device)
    else:
        # 估计数据大小并提示用户
        estimated_memory_gb = (len(train_formation_y) * len(train_dataset) * 4) / (1024 ** 3)
        print(f"估计需要内存: {estimated_memory_gb:.2f} GiB")
        
        # 确定批处理大小以适应GPU内存
        max_batch_size = min(1000, len(train_formation_y))
        print(f"使用批处理大小: {max_batch_size}")
        
        # 在CPU上创建结果张量，避免GPU内存不足
        train_matrix = torch.zeros((len(train_formation_y), len(train_dataset)), dtype=torch.float32)
        
        # 分批计算差异
        for i in range(0, len(train_formation_y), max_batch_size):
            end_idx = min(i + max_batch_size, len(train_formation_y))
            print(f"处理批次 {i}-{end_idx} / {len(train_formation_y)}")
            
            # 获取当前批次的形成能
            current_batch = train_formation_y[i:end_idx].cpu()
            
            # 在CPU上计算差异
            for j, data in enumerate(tqdm(current_batch, desc=f"批次 {i}-{end_idx} 形成能差异")):
                # 计算当前样本与所有前驱体能量总和的差异
                differences = data.item() - precursor_sums.cpu()
                train_matrix[i+j] = differences
        
        # 添加对角线掩码以排除自身
        diag_indices = torch.arange(len(train_dataset))
        train_matrix[diag_indices, diag_indices] = 100000  # 大值以排除自身
        
        print(f"保存训练集差异矩阵...")
        torch.save(train_matrix, f'./dataset/{args.split}/train_formation_energy_calculation_delta_G.pt')
        print(f"训练集差异计算完成，形状: {train_matrix.shape}")
    
    print(f"生成训练集检索结果...")
    # 分批处理检索结果生成
    make_retrieved('train', args.split, train_matrix.to(device), K, args.seed)

    # For Valid
    print("\n计算验证集形成能差异...")
    
    # 检查是否已存在差异文件
    if os.path.exists(f'./dataset/{args.split}/valid_formation_energy_calculation_delta_G.pt'):
        print(f"验证集差异文件已存在，跳过计算")
        valid_matrix = torch.load(f'./dataset/{args.split}/valid_formation_energy_calculation_delta_G.pt', map_location='cpu', weights_only=False)
        valid_matrix = valid_matrix.to(device)
    else:
        # 在CPU上创建结果张量
        valid_matrix = torch.zeros((len(valid_formation_y), len(train_dataset)), dtype=torch.float32)
        
        # 确定批处理大小
        max_batch_size = min(1000, len(valid_formation_y))
        
        # 分批计算差异
        for i in range(0, len(valid_formation_y), max_batch_size):
            end_idx = min(i + max_batch_size, len(valid_formation_y))
            print(f"处理批次 {i}-{end_idx} / {len(valid_formation_y)}")
            
            # 获取当前批次的形成能
            current_batch = valid_formation_y[i:end_idx].cpu()
            
            # 在CPU上计算差异
            for j, data in enumerate(tqdm(current_batch, desc=f"批次 {i}-{end_idx} 形成能差异")):
                differences = data.item() - precursor_sums.cpu()
                valid_matrix[i+j] = differences
        
        torch.save(valid_matrix, f'./dataset/{args.split}/valid_formation_energy_calculation_delta_G.pt')
        print(f"验证集差异计算完成，形状: {valid_matrix.shape}")
    
    print(f"生成验证集检索结果...")
    make_retrieved('valid', args.split, valid_matrix.to(device), K, args.seed)

    # For Test
    print("\n计算测试集形成能差异...")
    
    # 检查是否已存在差异文件
    if os.path.exists(f'./dataset/{args.split}/test_formation_energy_calculation_delta_G.pt'):
        print(f"测试集差异文件已存在，跳过计算")
        test_matrix = torch.load(f'./dataset/{args.split}/test_formation_energy_calculation_delta_G.pt', map_location='cpu', weights_only=False)
        test_matrix = test_matrix.to(device)
    else:
        # 在CPU上创建结果张量
        test_matrix = torch.zeros((len(test_formation_y), len(train_dataset)), dtype=torch.float32)
        
        # 确定批处理大小
        max_batch_size = min(1000, len(test_formation_y))
        
        # 分批计算差异
        for i in range(0, len(test_formation_y), max_batch_size):
            end_idx = min(i + max_batch_size, len(test_formation_y))
            print(f"处理批次 {i}-{end_idx} / {len(test_formation_y)}")
            
            # 获取当前批次的形成能
            current_batch = test_formation_y[i:end_idx].cpu()
            
            # 在CPU上计算差异
            for j, data in enumerate(tqdm(current_batch, desc=f"批次 {i}-{end_idx} 形成能差异")):
                differences = data.item() - precursor_sums.cpu()
                test_matrix[i+j] = differences
        
        torch.save(test_matrix, f'./dataset/{args.split}/test_formation_energy_calculation_delta_G.pt')
        print(f"测试集差异计算完成，形状: {test_matrix.shape}")
    
    print(f"生成测试集检索结果...")
    make_retrieved('test', args.split, test_matrix.to(device), K, args.seed)
    
    print(f"\n所有形成能差异计算和检索完成! 总用时: {time.time() - diff_start_time:.2f}秒")
    print(f"全部处理完成! 总用时: {time.time() - start_time:.2f}秒")

if __name__ == "__main__":
    main()