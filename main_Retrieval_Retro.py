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
    """将数据批次转换为模型需要的标准格式"""
    # 标准化批次中的每一项为元组
    for i, item in enumerate(batch):
        if isinstance(item, list) and len(item) == 3:
            batch[i] = tuple(item)
    
    # 批处理主图
    main_graphs = [item[0] for item in batch]
    batched_main_graphs = Batch.from_data_list(main_graphs)
    
    # 批处理第一组额外图 (MPC)
    first_additional_graphs = []
    for item in batch:
        for graph in item[1]:
            # 确保有precursor字段
            if not hasattr(graph, 'precursor') and hasattr(graph, 'y_lb_one'):
                graph.precursor = graph.y_lb_one
            first_additional_graphs.append(graph)
    
    # 如果没有额外图，创建填充图
    if not first_additional_graphs:
        from torch_geometric.data import Data
        dummy_graph = Data(
            x=torch.zeros((1, main_graphs[0].x.size(1))),
            edge_index=torch.zeros((2, 1), dtype=torch.long),
            edge_attr=torch.zeros((1, main_graphs[0].edge_attr.size(1))),
            fc_weight=torch.ones(1),
            precursor=torch.zeros(main_graphs[0].y_lb_one.shape) if hasattr(main_graphs[0], 'y_lb_one') else torch.zeros(798)
        )
        first_additional_graphs = [dummy_graph]
    
    batched_first_additional_graphs = Batch.from_data_list(first_additional_graphs)
    
    # 批处理第二组额外图 (NRE)
    second_additional_graphs = []
    for item in batch:
        for graph in item[2]:
            # 确保有precursor字段
            if not hasattr(graph, 'precursor') and hasattr(graph, 'y_lb_one'):
                graph.precursor = graph.y_lb_one
            second_additional_graphs.append(graph)
    
    # 如果没有额外图，创建填充图
    if not second_additional_graphs:
        from torch_geometric.data import Data
        dummy_graph = Data(
            x=torch.zeros((1, main_graphs[0].x.size(1))),
            edge_index=torch.zeros((2, 1), dtype=torch.long),
            edge_attr=torch.zeros((1, main_graphs[0].edge_attr.size(1))),
            fc_weight=torch.ones(1),
            precursor=torch.zeros(main_graphs[0].y_lb_one.shape) if hasattr(main_graphs[0], 'y_lb_one') else torch.zeros(798)
        )
        second_additional_graphs = [dummy_graph]
    
    batched_second_additional_graphs = Batch.from_data_list(second_additional_graphs)
    
    # 返回标准化的元组
    return (batched_main_graphs, batched_first_additional_graphs, batched_second_additional_graphs)

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


    args.device = 7
    # GPU setting
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)
    seed_everything(seed=args.seed)


    args.eval = 5
    args.retrieval = 'ours'
    args.split = 'year'
    args.embedder = 'Retrieval_Retro'

    print(f"加载数据集: {args.split}")
    train_file = f'./dataset/{args.split}/{args.split}_train_K_{args.K}.pt'
    valid_file = f'./dataset/{args.split}/{args.split}_valid_K_{args.K}.pt'
    test_file = f'./dataset/{args.split}/{args.split}_test_K_{args.K}.pt'
    
    # 检查K_{args.K}.pt文件是否存在
    if not os.path.exists(train_file):
        # 回退到标准格式
        print(f"未找到{train_file}，尝试替代文件")
        train_file = f'./dataset/{args.split}/train_K_{args.K}.pt'
        valid_file = f'./dataset/{args.split}/valid_K_{args.K}.pt'
        test_file = f'./dataset/{args.split}/test_K_{args.K}.pt'
        
        # 检查是否存在备用路径
        if not os.path.exists(train_file):
            # 再尝试一次
            print(f"未找到{train_file}，尝试最终替代路径")
            train_file = f'./dataset/{args.split}/train_final_mpc_nre_K_{args.K}.pt'
            valid_file = f'./dataset/{args.split}/valid_final_mpc_nre_K_{args.K}.pt'
            test_file = f'./dataset/{args.split}/test_final_mpc_nre_K_{args.K}.pt'
    
    print(f"训练集文件: {train_file}")
    print(f"验证集文件: {valid_file}")
    print(f"测试集文件: {test_file}")
    
    # 加载数据集
    train_dataset = torch.load(train_file, map_location=device, weights_only=False)
    valid_dataset = torch.load(valid_file, map_location=device, weights_only=False)
    test_dataset = torch.load(test_file, map_location=device, weights_only=False)
    
    # 标准化数据格式 - 无条件转换所有列表为元组
    if isinstance(train_dataset, list):
        # 统一转换为元组
        train_dataset = [tuple(x) if isinstance(x, list) else x for x in train_dataset]
        valid_dataset = [tuple(x) if isinstance(x, list) else x for x in valid_dataset]
        test_dataset = [tuple(x) if isinstance(x, list) else x for x in test_dataset]
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, collate_fn = custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size = 1, collate_fn = custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = 1, collate_fn = custom_collate_fn)

    print("数据集加载完成!")

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

    f = open(f"./experiments/Retrieval_Retro_{args.batch_size}_{args.retrieval}_{args.embedder}_{args.split}_{args.K}_result.txt", "a")

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
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()
        for bc, batch in enumerate(train_loader):

            y = batch[0].y_lb_one.reshape(len(batch[0].ptr)-1, -1)
            template_output = model(batch)
            loss_template = bce_loss(template_output, y)

            loss = loss_template
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

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
                        print(f'[FINAL]_MULTI: {best_state_multi}')
                        print(f'[FINAL]_MULTI: {best_state_recall}')
                        f.write("\n")
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
    print(f'[FINAL]_MULTI: {best_state_multi}')
    f.write("\n")
    f.write("Early stop!!\n")
    f.write(configuration)
    f.write(f"\nbest epoch : {best_epoch}")
    f.write(f"\nbest Top-1 ACC ONE: {multi_top_1_acc:.4f}")
    f.write(f"\nbest Top-3 ACC ONE: {multi_top_3_acc:.4f}")
    f.write(f"\nbest Top-5 ACC ONE: {multi_top_5_acc:.4f}")
    f.write(f"\nbest Top-10 ACC ONE: {multi_top_10_acc:.4f}")
    f.write(f"\nbest Micro Recall: {test_micro:.4f}")
    f.write(f"\nbest Macro Recall: {test_macro:.4f}")
    f.close()

if __name__ == "__main__":

    main()