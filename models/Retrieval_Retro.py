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
        # 直接处理输入数据，不进行任何类型检查和转换
        main_graph = data[0].to(self.device)
        additional_graph = data[1].to(self.device)
        additional_graph_2 = data[2].to(self.device)
        
        # 获取批次大小
        batch_size = len(main_graph.ptr) - 1 if hasattr(main_graph, 'ptr') else 1
        
        # 处理主图
        main_graph_x = self.gnn(main_graph)
        
        # 处理fc_weight
        if not hasattr(main_graph, 'fc_weight'):
            main_graph.fc_weight = torch.ones(main_graph.x.size(0), device=self.device)
        
        main_weighted_x = main_graph_x * main_graph.fc_weight.reshape(-1, 1)
        
        # 处理batch
        if not hasattr(main_graph, 'batch'):
            main_graph.batch = torch.zeros(main_graph.x.size(0), dtype=torch.long, device=self.device)
        
        main_graph_emb = scatter_sum(main_weighted_x, main_graph.batch, dim=0)
        
        # 处理额外图
        add_pooled = self.process_additional_graphs(additional_graph, batch_size)
        add_pooled_2 = self.process_additional_graphs(additional_graph_2, batch_size)
        
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
            
    def process_additional_graphs(self, additional_graph, batch_size):
        """处理额外图，无条件处理所有输入"""
        add_graph_outputs = []
        
        # 设置基本属性 - 无条件设置，确保接口一致
        if not hasattr(additional_graph, 'fc_weight'):
            additional_graph.fc_weight = torch.ones(additional_graph.x.size(0), device=self.device)
        
        if not hasattr(additional_graph, 'batch'):
            additional_graph.batch = torch.zeros(additional_graph.x.size(0), dtype=torch.long, device=self.device)
        
        # 生成图嵌入
        add_graph_x = self.gnn(additional_graph)
        add_weighted_x = add_graph_x * additional_graph.fc_weight.reshape(-1, 1)
        
        # 按批次分组处理
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
        
        # 确保输出长度与batch_size一致
        if len(add_graph_outputs) < batch_size:
            for i in range(batch_size - len(add_graph_outputs)):
                add_graph_outputs.append(torch.zeros((1, self.hidden_dim), device=self.device))
        elif len(add_graph_outputs) > batch_size:
            add_graph_outputs = add_graph_outputs[:batch_size]
            
        # 堆叠嵌入向量
        return torch.stack(add_graph_outputs, dim=0) 