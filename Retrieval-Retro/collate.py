import torch
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.loader import DataLoader
import json
import argparse
from tqdm import tqdm
import os
import sys
import time

# 设置命令行参数
parser = argparse.ArgumentParser(description='整合MPC和NRE检索结果')
parser.add_argument('--split', type=str, default='year', help='数据集拆分名称')
parser.add_argument('--mode', type=str, choices=['train', 'valid', 'test'], required=True, help='处理模式: train/valid/test')
parser.add_argument('--K', type=int, default=3, help='检索的候选数量')
parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
args = parser.parse_args()

def main():
    start_time = time.time()
    
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
    
    # 检查数据集目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集目录 {dataset_dir} 不存在")
        return
    
    # 检查输入文件是否存在
    mpc_retrieval_path = f"{dataset_dir}/{mode}_mpc_retrieved_{K}"
    nre_retrieval_path = f"{dataset_dir}/{mode}_nre_retrieved_{K}"
    
    if not os.path.exists(mpc_retrieval_path):
        print(f"错误: MPC检索结果不存在: {mpc_retrieval_path}")
        return
        
    if not os.path.exists(nre_retrieval_path):
        print(f"错误: NRE检索结果不存在: {nre_retrieval_path}")
        return
    
    # 检查训练数据集文件
    if not os.path.exists(train_mpc_path):
        print(f"错误: 训练数据集文件不存在: {train_mpc_path}")
        return
        
    if not os.path.exists(valid_mpc_path):
        print(f"错误: 验证数据集文件不存在: {valid_mpc_path}")
        return
        
    if not os.path.exists(test_mpc_path):
        print(f"错误: 测试数据集文件不存在: {test_mpc_path}")
        return
    
    print(f"加载数据集...")
    # 加载数据集
    try:
        print(f"加载训练集: {train_mpc_path}")
        train_dataset = torch.load(train_mpc_path, weights_only=False)
        print(f"加载验证集: {valid_mpc_path}")
        valid_dataset = torch.load(valid_mpc_path, weights_only=False)
        print(f"加载测试集: {test_mpc_path}")
        test_dataset = torch.load(test_mpc_path, weights_only=False)
        
        print(f"加载MPC检索结果: {mpc_retrieval_path}")
        with open(mpc_retrieval_path, "r") as f:
            candi_data = json.load(f)

        print(f"加载NRE检索结果: {nre_retrieval_path}")
        with open(nre_retrieval_path, "r") as f:
            candi_data_2 = json.load(f)
            
        # 打印数据集信息
        print(f"数据集大小: 训练集={len(train_dataset)}, 验证集={len(valid_dataset)}, 测试集={len(test_dataset)}")
        print(f"MPC检索结果数量: {len(candi_data)}")
        print(f"NRE检索结果数量: {len(candi_data_2)}")
            
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
    
    # 进度显示
    progress_bar = tqdm(total=len(dataset), desc=f"处理{mode}数据集")
    success_count = 0
    error_count = 0
    
    for idx, data in enumerate(dataset):
        progress_bar.update(1)
        
        try:
            # 确保索引存在于检索结果中
            idx_str = str(idx)
            if idx_str not in candi_data:
                print(f"警告: 索引 {idx} 在MPC检索结果中不存在")
                error_count += 1
                continue
                
            if idx_str not in candi_data_2:
                print(f"警告: 索引 {idx} 在NRE检索结果中不存在")
                error_count += 1
                continue
                
            candi_idx = candi_data[idx_str]
            candi_idx_2 = candi_data_2[idx_str]
            
            if not candi_idx or len(candi_idx) == 0:
                print(f"警告: 索引 {idx} 的MPC检索结果为空")
                error_count += 1
                continue
                
            if not candi_idx_2 or len(candi_idx_2) == 0:
                print(f"警告: 索引 {idx} 的NRE检索结果为空")
                error_count += 1
                continue
            
            # 创建子图容器 - 主图和两组额外图列表
            subgraph = []  # MPC检索结果
            subgraph_2 = []  # NRE检索结果
            
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
                    
                    # 创建检索图对象 - 与原始模型兼容
                    candi_graph = Data(
                        x=x, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        fc_weight=fc_weight, 
                        comp_fea=comp_fea, 
                        precursor=y
                    )
                    # 确保模型需要的属性存在
                    candi_graph.batch = torch.zeros(candi_graph.x.size(0), dtype=torch.long)
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
                    
                    # 创建检索图对象 - 与原始模型兼容
                    candi_graph2 = Data(
                        x=x, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        fc_weight=fc_weight, 
                        comp_fea=comp_fea, 
                        precursor=y
                    )
                    # 确保模型需要的属性存在
                    candi_graph2.batch = torch.zeros(candi_graph2.x.size(0), dtype=torch.long)
                    subgraph_2.append(candi_graph2)
                except Exception as e:
                    print(f"处理NRE候选 {i} 时出错: {e}")

            # 确保有检索结果
            if len(subgraph) == 0:
                print(f"警告: 索引 {idx} 的MPC检索结果处理后为空")
                error_count += 1
                continue
                
            if len(subgraph_2) == 0:
                print(f"警告: 索引 {idx} 的NRE检索结果处理后为空")
                error_count += 1
                continue

            # 创建原始模型期望的数据结构: (main_graph, [mpc_graphs], [nre_graphs])
            item = (data, subgraph, subgraph_2)
            new_data.append(item)
            success_count += 1
            
            # 每500个样本打印一次进度
            if (idx+1) % 500 == 0 or idx == len(dataset)-1:
                print(f"已处理 {idx+1}/{len(dataset)} 个样本，成功: {success_count}，失败: {error_count}")
                
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            error_count += 1
            continue
    
    progress_bar.close()
    
    # 保存整合结果
    if len(new_data) == 0:
        print(f"错误: 处理后没有有效数据，无法保存结果")
        return
        
    output_path = f"{dataset_dir}/{mode}_final_mpc_nre_K_{K}.pt"
    print(f"保存整合结果到: {output_path}, 共 {len(new_data)} 个样本")
    
    try:
        # 保存数据
        torch.save(new_data, output_path)
        print(f"{mode} 数据集整合完成!")
        
        # 验证保存的文件
        print(f"验证保存的文件...")
        test_load = torch.load(output_path)
        print(f"验证成功! 文件包含 {len(test_load)} 个样本")
        
        sample = test_load[0]
        print(f"数据结构验证:")
        print(f"- 类型: {type(sample)}")
        print(f"- 元素数量: {len(sample)}")
        print(f"- 主图节点特征: {sample[0].x.shape}")
        print(f"- MPC检索图数量: {len(sample[1])}")
        print(f"- NRE检索图数量: {len(sample[2])}")
        
    except Exception as e:
        print(f"保存或验证结果时出错: {e}")
        return
        
    print(f"处理完成! 总用时: {time.time() - start_time:.2f}秒")
    print(f"总共处理: {len(dataset)} 个样本")
    print(f"成功: {success_count} 个样本")
    print(f"失败: {error_count} 个样本")

if __name__ == "__main__":
    main()
                
