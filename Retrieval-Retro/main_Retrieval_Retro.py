import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys
from timeit import default_timer as timer
import time as local_time
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import scale
import json
from sklearn.model_selection import train_test_split
from models import Retrieval_Retro
import utils_main
from utils_main import recall_multilabel_multiple, top_k_acc_multiple

torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = "4"

from torch_geometric.data import Batch

def custom_collate_fn(batch):
    """自定义批次处理函数，用于处理包含主图和两组额外图的数据"""
    try:
        # 确保batch不为空
        if not batch:
            raise ValueError("空批次")
        
        # 批次结构检查
        for i, item in enumerate(batch):
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"批次项 {i} 不是一个包含3个元素的元组，而是 {type(item)}，长度: {len(item) if hasattr(item, '__len__') else 'N/A'}")
        
        # Batch main graphs
        main_graphs = []
        for i, item in enumerate(batch):
            if not hasattr(item[0], 'x') or not hasattr(item[0], 'edge_index'):
                raise ValueError(f"第 {i} 项的主图不是一个有效的PyG Data对象")
            main_graphs.append(item[0])
        
        batched_main_graphs = Batch.from_data_list(main_graphs)
        
        # Handle the first set of additional graphs
        first_additional_graphs = []
        for i, item in enumerate(batch):
            if not isinstance(item[1], list):
                raise ValueError(f"第 {i} 项的第一组额外图不是列表，而是 {type(item[1])}")
            
            for j, graph in enumerate(item[1]):
                if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                    raise ValueError(f"第 {i} 项的第一组额外图中的第 {j} 个图不是一个有效的PyG Data对象")
                first_additional_graphs.append(graph)
        
        # 确保有图形可以批处理
        if not first_additional_graphs:
            raise ValueError("第一组额外图为空")
            
        batched_first_additional_graphs = Batch.from_data_list(first_additional_graphs)

        # Handle the second set of additional graphs
        second_additional_graphs = []
        for i, item in enumerate(batch):
            if not isinstance(item[2], list):
                raise ValueError(f"第 {i} 项的第二组额外图不是列表，而是 {type(item[2])}")
                
            for j, graph in enumerate(item[2]):
                if not hasattr(graph, 'x') or not hasattr(graph, 'edge_index'):
                    raise ValueError(f"第 {i} 项的第二组额外图中的第 {j} 个图不是一个有效的PyG Data对象")
                second_additional_graphs.append(graph)
        
        # 确保有图形可以批处理
        if not second_additional_graphs:
            raise ValueError("第二组额外图为空")
            
        batched_second_additional_graphs = Batch.from_data_list(second_additional_graphs)
        
        # Return batched main graphs and both sets of batched additional graphs
        return batched_main_graphs, batched_first_additional_graphs, batched_second_additional_graphs
        
    except Exception as e:
        print(f"批次处理错误: {e}")
        print(f"批次类型: {type(batch)}, 批次长度: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
        if batch and hasattr(batch, '__getitem__'):
            print(f"批次第一项类型: {type(batch[0])}")
            if isinstance(batch[0], tuple) and len(batch[0]) >= 3:
                print(f"  主图类型: {type(batch[0][0])}")
                print(f"  第一组额外图类型: {type(batch[0][1])}")
                print(f"  第二组额外图类型: {type(batch[0][2])}")
                
                # 检查第一组额外图的详细信息
                if isinstance(batch[0][1], list) and batch[0][1]:
                    print(f"  第一组额外图第一项类型: {type(batch[0][1][0])}")
                    
                # 检查第二组额外图的详细信息
                if isinstance(batch[0][2], list) and batch[0][2]:
                    print(f"  第二组额外图第一项类型: {type(batch[0][2][0])}")
        
        # 如果无法正常处理，则返回一个空批次标记
        # 主程序需要处理这种情况
        return None

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

def main():
    args = utils_main.parse_args()
    train_config = utils_main.training_config(args)
    configuration = utils_main.exp_get_name_RetroPLEX(train_config)
    print(f'configuration: {configuration}')

    # 使用命令行参数中的设备ID
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(f"使用设备: {device}")
    print(f"数据集拆分: {args.split}")
    print(f"检索数量K: {args.K}")
    
    seed_everything(seed=args.seed)

    args.eval = 5
    args.retrieval = 'ours'
    args.embedder = 'Retrieval_Retro'

    # 修改数据加载路径，使用args.split而不是硬编码"year"
    print(f"加载数据集: {args.split}")
    train_file = f'./dataset/{args.split}/train_K_{args.K}.pt'
    valid_file = f'./dataset/{args.split}/valid_K_{args.K}.pt'
    test_file = f'./dataset/{args.split}/test_K_{args.K}.pt'
    
    print(f"训练集文件: {train_file}")
    print(f"验证集文件: {valid_file}")
    print(f"测试集文件: {test_file}")
    
    try:
        train_dataset = torch.load(train_file, map_location=device, weights_only=False)
        valid_dataset = torch.load(valid_file, map_location=device, weights_only=False)
        test_dataset = torch.load(test_file, map_location=device, weights_only=False)
    except Exception as e:
        print(f"错误: 无法加载数据集文件: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, collate_fn = custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size = 1, collate_fn = custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = 1, collate_fn = custom_collate_fn)

    print("数据集加载完成!")
    print(f"数据集大小: 训练集={len(train_dataset)}, 验证集={len(valid_dataset)}, 测试集={len(test_dataset)}")

    # 只在开始时打印一次数据维度检查，避免冗余输出
    print("\n==== 数据维度检查 ====")
    try:
        # 检查第一个训练数据点
        train_sample = train_dataset[0]
        print(f"训练样本主图特征维度: x={train_sample[0].x.shape}, edge_attr={train_sample[0].edge_attr.shape}")
        print(f"训练样本MPC检索图数量: {len(train_sample[1])}")
        print(f"训练样本NRE检索图数量: {len(train_sample[2])}")
        
        # 检查一个MPC检索图和NRE检索图
        if len(train_sample[1]) > 0:
            print(f"MPC检索图特征维度: x={train_sample[1][0].x.shape}, edge_attr={train_sample[1][0].edge_attr.shape}")
        if len(train_sample[2]) > 0:
            print(f"NRE检索图特征维度: x={train_sample[2][0].x.shape}, edge_attr={train_sample[2][0].edge_attr.shape}")
    except Exception as e:
        print(f"数据检查时出错: {e}")

    gnn = args.gnn
    layers = args.layers
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    n_bond_feat = train_dataset[0][0].edge_attr.shape[1]
    output_dim = train_dataset[0][0].y_multiple.shape[1] 
    embedder = args.embedder
    num_heads = args.num_heads
    t_layers = args.t_layers
    t_layers_sa = args.t_layers_sa
    thres = 'normal'

    # 在结果文件名中添加数据集标识
    result_file = f"./experiments/Retrieval_Retro_{args.batch_size}_{args.retrieval}_{args.embedder}_{args.split}_{args.K}_result.txt"
    print(f"结果将保存到: {result_file}")
    
    # 确保experiments目录存在
    os.makedirs("./experiments", exist_ok=True)
    
    f = open(result_file, "a")

    # 打印模型配置
    print("\n==== 模型配置 ====")
    print(f"GNN: {gnn}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"输入维度: {input_dim}")
    print(f"边特征维度: {n_bond_feat}")
    print(f"输出维度: {output_dim}")
    print(f"网络层数: {layers}")
    print(f"Transformer层数: {t_layers}")
    print(f"Self-Attention层数: {t_layers_sa}")
    print(f"注意力头数: {num_heads}")

    if embedder == 'Retrieval_Retro': 
        model = Retrieval_Retro(gnn, layers, input_dim, output_dim, hidden_dim, n_bond_feat, device, t_layers, t_layers_sa, num_heads).to(device)
    else:
        print("############### Wrong Model Name ################")

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)
    bce_loss = nn.BCELoss()

    train_loss = 0
    num_batch = int(len(train_dataset)/args.batch_size)
    best_acc = 0
    best_epoch = 0
    test_macro = 0
    test_micro = 0
    best_acc_list = []
    
    # 只在第一个epoch的第一个batch进行模型调试
    debug_first_batch = True
    
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()
        for bc, batch in enumerate(train_loader):
            # 检查批次是否有效
            if batch is None:
                print(f"跳过无效批次 {bc}")
                continue
            
            # 只在第一个epoch的第一个batch进行模型调试
            if debug_first_batch and epoch == 0 and bc == 0:
                try:
                    print(f"\n[批次调试] 批次{bc}:")
                    print(f"主图节点数量: {batch[0].x.size(0)}")
                    print(f"主图节点特征维度: {batch[0].x.shape}")
                    print(f"主图批次大小: {len(batch[0].ptr)-1}")
                    
                    # 检查MPC检索结果
                    mpc_graph = batch[1]
                    # 检查mpc_graph是否为PyG数据对象，而不是列表
                    if hasattr(mpc_graph, 'x'):
                        print(f"MPC图总数: {mpc_graph.x.size(0)}")
                        print(f"MPC图节点特征维度: {mpc_graph.x.shape}")
                    else:
                        print(f"警告: MPC图不是预期的数据结构，是 {type(mpc_graph)}，跳过检查")
                    
                    # 检查NRE检索结果
                    nre_graph = batch[2]
                    # 检查nre_graph是否为PyG数据对象，而不是列表
                    if hasattr(nre_graph, 'x'):
                        print(f"NRE图总数: {nre_graph.x.size(0)}")
                        print(f"NRE图节点特征维度: {nre_graph.x.shape}")
                    else:
                        print(f"警告: NRE图不是预期的数据结构，是 {type(nre_graph)}，跳过检查")
                    
                    # 打印hidden_dim，确认其值
                    print(f"模型hidden_dim: {hidden_dim}")
                    print(f"模型fusion_linear层输入预期维度: {hidden_dim*2}")
                    print(f"模型fusion_linear层输出预期维度: {hidden_dim}")
                    debug_first_batch = False
                except Exception as e:
                    print(f"批次检查时出错: {e}")
                    print(f"错误类型: {type(e).__name__}")
                    # 打印更多信息帮助调试
                    print(f"主图类型: {type(batch[0])}")
                    print(f"MPC图类型: {type(batch[1])}")
                    print(f"NRE图类型: {type(batch[2])}")
                    debug_first_batch = False

            # 检查是否有必要的属性
            try:
                y = batch[0].y_lb_one.reshape(len(batch[0].ptr)-1, -1)
            except AttributeError as e:
                print(f"批次{bc}缺少必要属性: {e}")
                continue
            
            # 减少冗余输出，只在出错时显示详细信息
            try:
                template_output = model(batch)
            except Exception as e:
                print(f"\n模型前向传播中出现错误: {e}")
                print(f"错误类型: {type(e).__name__}")
                # 尝试进一步分析错误
                try:
                    # 检查模型各组件输入维度
                    main_graph = batch[0].to(device)
                    
                    # 检查additional_graph和additional_graph_2的类型
                    additional_graph = batch[1]
                    additional_graph_2 = batch[2]
                    
                    print(f"额外图类型检查:")
                    print(f"additional_graph类型: {type(additional_graph)}")
                    print(f"additional_graph_2类型: {type(additional_graph_2)}")
                    
                    # 如果是列表，打印第一个元素类型
                    if isinstance(additional_graph, list) and len(additional_graph) > 0:
                        print(f"additional_graph[0]类型: {type(additional_graph[0])}")
                    if isinstance(additional_graph_2, list) and len(additional_graph_2) > 0:
                        print(f"additional_graph_2[0]类型: {type(additional_graph_2[0])}")
                    
                    # GNN处理
                    main_graph_x = model.gnn(main_graph)
                    print(f"主图GNN输出维度: {main_graph_x.shape}")
                    
                    main_weighted_x = main_graph_x * main_graph.fc_weight.reshape(-1, 1)
                    main_graph_emb = torch.scatter_add(main_weighted_x, dim=0, index=main_graph.batch.view(-1, 1).repeat(1, main_weighted_x.size(1)))
                    print(f"主图嵌入维度: {main_graph_emb.shape}")
                    
                    # 谨慎检查检索图类型
                    if isinstance(additional_graph, list) and len(additional_graph) > 0:
                        add_graph = additional_graph[0]
                        # 检查图是否为PyG数据对象
                        if hasattr(add_graph, 'x') and hasattr(add_graph, 'edge_index'):
                            add_graph = add_graph.to(device)
                            add_graph_x = model.gnn(add_graph)
                            print(f"检索图GNN输出维度: {add_graph_x.shape}")
                        else:
                            print(f"检索图数据类型不正确: {type(add_graph)}")
                except Exception as nested_e:
                    print(f"无法分析错误: {nested_e}")
                    print(f"嵌套错误类型: {type(nested_e).__name__}")
                
                # 终止当前批次，尝试下一个批次
                continue
                
            loss_template = bce_loss(template_output, y)
            loss = loss_template
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

            # 减少输出频率，只每10个批次更新一次进度
            if bc % 10 == 0 or bc == num_batch - 1:
                sys.stdout.write(f'\r[ epoch {epoch+1}/{args.epochs} | batch {bc}/{num_batch} ] Total Loss : {(train_loss/(bc+1)):.4f}')
                sys.stdout.flush()

        if (epoch + 1) % args.eval == 0 :
            model.eval()
            multi_val_top_1_list = []
            multi_val_top_3_list = []
            multi_val_top_5_list = []
            multi_val_top_10_list = []

            val_micro_rec_list = []
            val_macro_rec_list = []

            with torch.no_grad():

                for bc, batch in enumerate(valid_loader):

                    template_output = model(batch)
                    assert batch[0].y_multiple_len.sum().item() == batch[0].y_multiple.size(0)

                    absolute_indices = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch[0].y_multiple_len, dim=0)])
                    split_tensors = [batch[0].y_multiple[start:end] for start, end in zip(absolute_indices[:-1], absolute_indices[1:])]

                    multi_label = batch[0].y_multiple

                    # Top-K Accuracy
                    multi_val_top_1_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 1)
                    multi_val_top_1_list.append(multi_val_top_1_scores)
                    multi_val_top_3_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 3)
                    multi_val_top_3_list.append(multi_val_top_3_scores)
                    multi_val_top_5_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 5)
                    multi_val_top_5_list.append(multi_val_top_5_scores)
                    multi_val_top_10_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 10)
                    multi_val_top_10_list.append(multi_val_top_10_scores)

                    # Macro Recall/Micro Recall
                    val_micro_rec, val_macro_rec = recall_multilabel_multiple(split_tensors, template_output.detach().cpu().numpy(), threshold= thres)
                    val_macro_rec_list.append(val_macro_rec)
                    val_micro_rec_list.append(val_micro_rec)

                multi_val_top_1_acc = np.mean(np.concatenate(multi_val_top_1_list))
                multi_val_top_3_acc = np.mean(np.concatenate(multi_val_top_3_list))
                multi_val_top_5_acc = np.mean(np.concatenate(multi_val_top_5_list))
                multi_val_top_10_acc = np.mean(np.concatenate(multi_val_top_10_list))

                val_micro = np.mean(np.concatenate(val_micro_rec_list))
                val_macro = np.mean(np.concatenate(val_macro_rec_list))

                print(f'\n Valid_multi | Epoch: {epoch+1} | Top-1 ACC: {multi_val_top_1_acc:.4f} | Top-3 ACC: {multi_val_top_3_acc:.4f} | Top-5 ACC: {multi_val_top_5_acc:.4f} | Top-10 ACC: {multi_val_top_10_acc:.4f} ')
                print(f'\n Valid Recall | Epoch: {epoch+1} | Micro_Recall: {val_micro:.4f} | Macro_Recall: {val_macro:.4f} ')


                if multi_val_top_5_acc > best_acc:
                    best_acc = multi_val_top_5_acc
                    best_epoch = epoch + 1

                    model.eval()

                    multi_top_1_list = []
                    multi_top_3_list = []
                    multi_top_5_list = []
                    multi_top_10_list = []

                    test_micro_rec_list = []
                    test_macro_rec_list = []

                    with torch.no_grad():
                        for bc, batch in enumerate(test_loader):

                            template_output = model(batch)

                            assert batch[0].y_multiple_len.sum().item() == batch[0].y_multiple.size(0)

                            absolute_indices = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch[0].y_multiple_len, dim=0)])
                            split_tensors = [batch[0].y_multiple[start:end] for start, end in zip(absolute_indices[:-1], absolute_indices[1:])]

                            multi_label = batch[0].y_multiple

                            multi_top_1_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 1)
                            multi_top_1_list.append(multi_top_1_scores)
                            multi_top_3_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 3)
                            multi_top_3_list.append(multi_top_3_scores)
                            multi_top_5_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 5)
                            multi_top_5_list.append(multi_top_5_scores)
                            multi_top_10_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 10)
                            multi_top_10_list.append(multi_top_10_scores)

                            # Macro Recall/Micro Recall
                            test_micro_rec, test_macro_rec = recall_multilabel_multiple(split_tensors, template_output.detach().cpu().numpy(), threshold=thres)
                            test_macro_rec_list.append(test_macro_rec)
                            test_micro_rec_list.append(test_micro_rec)

                        multi_top_1_acc = np.mean(np.concatenate(multi_top_1_list))
                        multi_top_3_acc = np.mean(np.concatenate(multi_top_3_list))
                        multi_top_5_acc = np.mean(np.concatenate(multi_top_5_list))
                        multi_top_10_acc = np.mean(np.concatenate(multi_top_10_list))

                        test_micro = np.mean(np.concatenate(test_micro_rec_list))
                        test_macro = np.mean(np.concatenate(test_macro_rec_list))

                        print(f'\n Test_multi | Epoch: {epoch+1} | Top-1 ACC: {multi_top_1_acc:.4f} | Top-3 ACC: {multi_top_3_acc:.4f} | Top-5 ACC: {multi_top_5_acc:.4f} | Top-10 ACC: {multi_top_10_acc:.4f} ')
                        print(f'\n Test Recall | Epoch: {epoch+1} | Micro_Recall: {test_micro:.4f} | Macro_Recall: {test_macro:.4f} ')


                best_acc_list.append(multi_top_5_acc)
                best_state_multi = f'[Best epoch: {best_epoch}] | Top-1 ACC: {multi_top_1_acc:.4f} | Top-3 ACC: {multi_top_3_acc:.4f} | Top-5 ACC: {multi_top_5_acc:.4f} | Top-10 ACC: {multi_top_10_acc:.4f}'
                best_state_recall = f'[Best epoch: {best_epoch}] | Micro Recall: {test_micro:.4f} | Macro Recall: {test_macro:.4f}'

                if len(best_acc_list) > int(args.es / args.eval):
                    if best_acc_list[-1] == best_acc_list[-int(args.es / args.eval)]:
                        print(f'!!Early Stop!!')
                        print(f'[数据集: {args.split}] [FINAL]_MULTI: {best_state_multi}')
                        print(f'[数据集: {args.split}] [FINAL]_MULTI: {best_state_recall}')
                        f.write("\n")
                        f.write(f"数据集: {args.split}\n")
                        f.write("Early stop!!\n")
                        f.write(configuration)
                        f.write(f"\nbest epoch : {best_epoch}")
                        f.write(f"\nbest Top-1 ACC  MULTI: {multi_top_1_acc:.4f}")
                        f.write(f"\nbest Top-3 ACC MULTI: {multi_top_3_acc:.4f}")
                        f.write(f"\nbest Top-5 ACC MULTI: {multi_top_5_acc:.4f}")
                        f.write(f"\nbest Top-10 ACC MULTI: {multi_top_10_acc:.4f}")
                        f.write(f"\nbest Micro Recall: {test_micro:.4f}")
                        f.write(f"\nbest Macro Recall: {test_macro:.4f}")
                        sys.exit()


    print(f'Training Done not early stopping')
    print(f'[数据集: {args.split}] [FINAL]_MULTI: {best_state_multi}')
    f.write("\n")
    f.write(f"数据集: {args.split}\n")
    f.write("Not early stopping!!\n")
    f.write(configuration)
    f.write(f"\nbest epoch : {best_epoch}")
    f.write(f"\nbest Top-1 ACC  MULTI: {multi_top_1_acc:.4f}")
    f.write(f"\nbest Top-3 ACC MULTI: {multi_top_3_acc:.4f}")
    f.write(f"\nbest Top-5 ACC MULTI: {multi_top_5_acc:.4f}")
    f.write(f"\nbest Top-10 ACC MULTI: {multi_top_10_acc:.4f}")
    f.write(f"\nbest Micro Recall: {test_micro:.4f}")
    f.write(f"\nbest Macro Recall: {test_macro:.4f}")
    f.close()

if __name__ == "__main__":
    main()