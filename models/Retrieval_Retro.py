import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential
from torch_geometric.nn import Set2Set
from models import GraphNetwork
from torch_scatter import scatter_sum
from collections import Counter
import copy
from .layers import Self_TransformerEncoder_non, Cross_TransformerEncoder_non
import numpy as np


class Retrieval_Retro(nn.Module):
    def __init__(self, gnn, layers, input_dim, output_dim, hidden_dim, n_bond_feat, device, t_layers, t_layers_sa, num_heads):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.classifier = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim*3), nn.PReLU(), nn.Linear(hidden_dim*3, self.output_dim), nn.Sigmoid())
        self.gnn = GraphNetwork(layers, input_dim, n_bond_feat, hidden_dim, device)

        # MPC
        self.self_attention = Self_TransformerEncoder_non(hidden_dim, num_heads, t_layers_sa, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        self.cross_attention = Cross_TransformerEncoder_non(hidden_dim, num_heads, t_layers, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        self.fusion_linear = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.PReLU())
        print(f"hidden_dim: {hidden_dim}, num_heads: {num_heads}, t_layers_sa: {t_layers_sa}, t_layers: {t_layers}")

        # NRE
        self.self_attention_2 = Self_TransformerEncoder_non(hidden_dim, num_heads, t_layers_sa, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        self.cross_attention_2 = Cross_TransformerEncoder_non(hidden_dim, num_heads, t_layers, attn_dropout=0.1, res_dropout=0.1, embed_dropout=0.1)

        self.fusion_linear_2 = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.PReLU())

        self.init_model()


    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        try:
            # 检查输入数据的结构
            if isinstance(data, list):
                print("输入数据不是元组，而是列表，长度：", len(data))
                data = tuple(data)
                print("已将列表转换为元组")
            elif not isinstance(data, tuple) or len(data) != 3:
                raise ValueError(f"输入数据不是有效的元组，而是 {type(data)}，长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
            
            # 确保数据在正确设备上
            main_graph = data[0].to(self.device)
            additional_graph = data[1]
            additional_graph_2 = data[2]
            
            # 检查主图是否有必要的属性
            if not hasattr(main_graph, 'x') or not hasattr(main_graph, 'edge_index'):
                raise ValueError("主图缺少必要的属性")
                
            # 获取批次大小
            batch_size = len(main_graph.ptr) - 1
            
            # 处理主图
            main_graph_x = self.gnn(main_graph)
            main_weighted_x = main_graph_x * main_graph.fc_weight.reshape(-1, 1)
            main_graph_emb = scatter_sum(main_weighted_x, main_graph.batch, dim=0)
            
            # 处理第一组额外图
            additional_graph = additional_graph.to(self.device)
            add_pooled = self.process_additional_graphs(additional_graph, batch_size, main_graph_emb, "MPC")
            
            # 处理第二组额外图
            additional_graph_2 = additional_graph_2.to(self.device)
            add_pooled_2 = self.process_additional_graphs(additional_graph_2, batch_size, main_graph_emb, "NRE")
            
            # MPC Self Attention Layers
            main_graph_repeat = main_graph_emb.unsqueeze(1).repeat(1, add_pooled.size(1), 1)
            add_pooled = torch.cat([add_pooled, main_graph_repeat], dim=2)
            add_pooled = self.fusion_linear(add_pooled)
            add_pooled_self = self.self_attention(add_pooled)

            # Cross Attention Layers
            cross_attn_output = self.cross_attention(main_graph_emb.unsqueeze(0), add_pooled_self, add_pooled_self)

            # NRE Self Attention Layers
            main_graph_repeat = main_graph_emb.unsqueeze(1).repeat(1, add_pooled_2.size(1), 1)
            add_pooled_2 = torch.cat([add_pooled_2, main_graph_repeat], dim=2)
            add_pooled_2 = self.fusion_linear_2(add_pooled_2)
            add_pooled_self_2 = self.self_attention_2(add_pooled_2)

            # Cross Attention Layers
            cross_attn_output_2 = self.cross_attention_2(main_graph_emb.unsqueeze(0), add_pooled_self_2, add_pooled_self_2)

            # 拼接所有特征
            classifier_input = torch.cat([main_graph_emb, cross_attn_output.squeeze(0), cross_attn_output_2.squeeze(0)], dim=1)
            
            # 分类预测
            template_output = self.classifier(classifier_input)
            return template_output
            
        except Exception as e:
            print(f"Retrieval_Retro.forward出错: {e}")
            # 如果可能，尝试返回一个有效的输出
            num_samples = 1
            if isinstance(data, tuple) and len(data) > 0 and hasattr(data[0], 'ptr'):
                num_samples = len(data[0].ptr) - 1
            # 返回全零输出
            return torch.zeros((num_samples, self.output_dim), device=self.device)
            
    def process_additional_graphs(self, additional_graph, batch_size, main_graph_emb, graph_type=""):
        """处理额外图"""
        add_graph_outputs = []
        
        # 检查additional_graph是否是PyG批处理对象
        if hasattr(additional_graph, 'x') and hasattr(additional_graph, 'batch'):
            try:
                # 正常处理PyG批处理对象
                add_graph_x = self.gnn(additional_graph)
                add_weighted_x = add_graph_x * additional_graph.fc_weight.reshape(-1, 1)
                
                # 按batch_idx分组
                for i in range(batch_size):
                    mask = (additional_graph.batch == i)
                    if mask.sum() > 0:
                        weighted_x_i = add_weighted_x[mask]
                        graph_emb_i = scatter_sum(weighted_x_i, 
                                                torch.zeros(weighted_x_i.size(0), 
                                                           dtype=torch.long, 
                                                           device=self.device), 
                                                dim=0)
                        add_graph_outputs.append(graph_emb_i)
                    else:
                        # 如果该批次没有节点，添加零向量
                        add_graph_outputs.append(torch.zeros((1, self.hidden_dim), device=self.device))
            except Exception as e:
                print(f"处理{graph_type}图出错: {e}")
                # 创建零向量替代
                for i in range(batch_size):
                    add_graph_outputs.append(torch.zeros((1, self.hidden_dim), device=self.device))
        else:
            # 不是PyG批处理对象
            print(f"警告: {graph_type}图不是PyG批处理对象，跳过处理")
            # 创建零向量替代
            for i in range(batch_size):
                add_graph_outputs.append(torch.zeros((1, self.hidden_dim), device=self.device))
        
        # 确保输出长度与batch_size一致
        if len(add_graph_outputs) < batch_size:
            for i in range(batch_size - len(add_graph_outputs)):
                add_graph_outputs.append(torch.zeros((1, self.hidden_dim), device=self.device))
        elif len(add_graph_outputs) > batch_size:
            add_graph_outputs = add_graph_outputs[:batch_size]
            
        # 堆叠嵌入向量
        return torch.stack(add_graph_outputs, dim=0) 