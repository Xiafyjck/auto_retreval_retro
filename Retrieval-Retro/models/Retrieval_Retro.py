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
            if not isinstance(data, tuple) or len(data) != 3:
                print(f"输入数据不是有效的元组，而是 {type(data)}，长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                # 如果是列表而不是元组，尝试转换
                if isinstance(data, list) and len(data) == 3:
                    data = tuple(data)
                    print("已将列表转换为元组")
                else:
                    raise ValueError(f"输入数据不是有效的元组或列表，无法处理")
            
            main_graph = data[0].to(self.device)
            additional_graph = data[1]
            additional_graph_2 = data[2]
            
            # 检查主图是否有必要的属性
            if not hasattr(main_graph, 'x') or not hasattr(main_graph, 'edge_index'):
                raise ValueError("主图缺少必要的属性")
                
            # 获取批次大小
            batch_size = len(main_graph.ptr) - 1
            # 只在第一次前向传播时打印批次大小
            if not hasattr(self, '_printed_batch_size'):
                print(f"批次大小: {batch_size}")
                self._printed_batch_size = True

            main_graph_x = self.gnn(main_graph)
            main_weighted_x = main_graph_x * main_graph.fc_weight.reshape(-1, 1)
            main_graph_emb = scatter_sum(main_weighted_x, main_graph.batch, dim = 0)
            
            # 确保main_graph_emb的第一维是batch_size
            if main_graph_emb.size(0) != batch_size:
                print(f"警告: 主图嵌入维度不匹配批次大小! 嵌入: {main_graph_emb.size(0)}, 批次: {batch_size}")
                if main_graph_emb.size(0) < batch_size:
                    # 扩展嵌入以匹配批次大小
                    repeat_times = batch_size // main_graph_emb.size(0)
                    remainder = batch_size % main_graph_emb.size(0)
                    expanded_emb = main_graph_emb.repeat(repeat_times, 1)
                    if remainder > 0:
                        expanded_emb = torch.cat([expanded_emb, main_graph_emb[:remainder]], dim=0)
                    main_graph_emb = expanded_emb
                else:
                    # 截取嵌入以匹配批次大小
                    main_graph_emb = main_graph_emb[:batch_size]

            # 处理第一组额外图
            add_graph_outputs = []
            
            # 检查additional_graph是否是PyG批处理对象
            if hasattr(additional_graph, 'x') and hasattr(additional_graph, 'batch'):
                # 计算每个图的边界索引
                batch_sizes = torch.bincount(additional_graph.batch).tolist()
                cum_sizes = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(batch_sizes), dim=0)])
                
                # 分别处理每个图
                for i in range(len(batch_sizes)):
                    start_idx = cum_sizes[i]
                    end_idx = cum_sizes[i+1]
                    
                    # 创建当前图的子图
                    sub_x = additional_graph.x[start_idx:end_idx]
                    sub_edge_index = additional_graph.edge_index[:, additional_graph.edge_index[0] >= start_idx]
                    sub_edge_index = sub_edge_index[:, sub_edge_index[0] < end_idx]
                    sub_edge_index = sub_edge_index - start_idx
                    
                    sub_edge_attr = additional_graph.edge_attr[sub_edge_index[0].long()]
                    sub_fc_weight = additional_graph.fc_weight[start_idx:end_idx]
                    
                    # 处理子图
                    from torch_geometric.data import Data
                    sub_graph = Data(
                        x=sub_x, 
                        edge_index=sub_edge_index, 
                        edge_attr=sub_edge_attr,
                        fc_weight=sub_fc_weight
                    ).to(self.device)
                    
                    sub_graph_x = self.gnn(sub_graph)
                    sub_weighted_x = sub_graph_x * sub_graph.fc_weight.reshape(-1, 1)
                    sub_graph_emb = scatter_sum(sub_weighted_x, torch.zeros(len(sub_graph.x), dtype=torch.long, device=self.device), dim=0)
                    add_graph_outputs.append(sub_graph_emb)
            else:
                # 如果不是PyG批处理对象，则应该是批处理前的单独图列表
                print(f"警告: additional_graph不是PyG批处理对象，跳过处理")
                # 添加一个空占位符
                dummy_emb = torch.zeros((main_graph_emb.size(0), self.hidden_dim), device=self.device)
                add_graph_outputs.append(dummy_emb)
            
            # 如果没有有效的额外图，则创建一个全零张量
            if not add_graph_outputs:
                add_pooled = torch.zeros((main_graph_emb.size(0), 1, self.hidden_dim), device=self.device)
            else:
                add_pooled = torch.stack(add_graph_outputs, dim=1)
            
            # 处理第二组额外图
            add_graph_outputs_2 = []
            
            # 检查additional_graph_2是否是PyG批处理对象
            if hasattr(additional_graph_2, 'x') and hasattr(additional_graph_2, 'batch'):
                # 计算每个图的边界索引
                batch_sizes = torch.bincount(additional_graph_2.batch).tolist()
                cum_sizes = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(batch_sizes), dim=0)])
                
                # 分别处理每个图
                for i in range(len(batch_sizes)):
                    start_idx = cum_sizes[i]
                    end_idx = cum_sizes[i+1]
                    
                    # 创建当前图的子图
                    sub_x = additional_graph_2.x[start_idx:end_idx]
                    sub_edge_index = additional_graph_2.edge_index[:, additional_graph_2.edge_index[0] >= start_idx]
                    sub_edge_index = sub_edge_index[:, sub_edge_index[0] < end_idx]
                    sub_edge_index = sub_edge_index - start_idx
                    
                    sub_edge_attr = additional_graph_2.edge_attr[sub_edge_index[0].long()]
                    sub_fc_weight = additional_graph_2.fc_weight[start_idx:end_idx]
                    
                    # 处理子图
                    from torch_geometric.data import Data
                    sub_graph = Data(
                        x=sub_x, 
                        edge_index=sub_edge_index, 
                        edge_attr=sub_edge_attr,
                        fc_weight=sub_fc_weight
                    ).to(self.device)
                    
                    sub_graph_x = self.gnn(sub_graph)
                    sub_weighted_x = sub_graph_x * sub_graph.fc_weight.reshape(-1, 1)
                    sub_graph_emb = scatter_sum(sub_weighted_x, torch.zeros(len(sub_graph.x), dtype=torch.long, device=self.device), dim=0)
                    add_graph_outputs_2.append(sub_graph_emb)
            else:
                # 如果不是PyG批处理对象，则应该是批处理前的单独图列表
                print(f"警告: additional_graph_2不是PyG批处理对象，跳过处理")
                # 添加一个空占位符
                dummy_emb = torch.zeros((main_graph_emb.size(0), self.hidden_dim), device=self.device)
                add_graph_outputs_2.append(dummy_emb)
            
            # 如果没有有效的额外图，则创建一个全零张量
            if not add_graph_outputs_2:
                add_pooled_2 = torch.zeros((main_graph_emb.size(0), 1, self.hidden_dim), device=self.device)
            else:
                add_pooled_2 = torch.stack(add_graph_outputs_2, dim=1)
                
            # MPC
            # Self Attention Layers
            main_graph_repeat = main_graph_emb.unsqueeze(1).repeat(1, add_pooled.shape[1], 1).to(self.device)
            add_pooled = torch.cat([add_pooled, main_graph_repeat], dim=2)
            add_pooled = self.fusion_linear(add_pooled)

            add_pooled_self = self.self_attention(add_pooled)

            # Cross Attention Layers
            cross_attn_output = self.cross_attention(main_graph_emb.unsqueeze(0), add_pooled_self, add_pooled_self)

            # NRE
            # Self Attention Layers
            main_graph_repeat = main_graph_emb.unsqueeze(1).repeat(1, add_pooled_2.shape[1], 1).to(self.device)
            add_pooled_2 = torch.cat([add_pooled_2, main_graph_repeat], dim=2)
            add_pooled_2 = self.fusion_linear_2(add_pooled_2)

            add_pooled_self_2 = self.self_attention_2(add_pooled_2)

            # Cross Attention Layers
            cross_attn_output_2 = self.cross_attention_2(main_graph_emb.unsqueeze(0), add_pooled_self_2, add_pooled_self_2)

            # 拼接所有特征
            classifier_input = torch.cat([main_graph_emb, cross_attn_output.squeeze(0), cross_attn_output_2.squeeze(0)], dim=1).to(self.device)
            
            # 确保classifier_input的第一维是batch_size
            if classifier_input.size(0) != batch_size:
                print(f"警告: 分类器输入维度不匹配批次大小! 输入: {classifier_input.size(0)}, 批次: {batch_size}")
                if classifier_input.size(0) < batch_size:
                    # 扩展输入以匹配批次大小
                    repeat_times = batch_size // classifier_input.size(0)
                    remainder = batch_size % classifier_input.size(0)
                    expanded_input = classifier_input.repeat(repeat_times, 1)
                    if remainder > 0:
                        expanded_input = torch.cat([expanded_input, classifier_input[:remainder]], dim=0)
                    classifier_input = expanded_input
                else:
                    # 截取输入以匹配批次大小
                    classifier_input = classifier_input[:batch_size]
            
            template_output = self.classifier(classifier_input)
            
            # 最后再次检查输出的batch大小
            if template_output.size(0) != batch_size:
                print(f"警告: 最终输出维度不匹配批次大小! 输出: {template_output.size(0)}, 批次: {batch_size}")
                if template_output.size(0) < batch_size:
                    # 扩展输出以匹配批次大小
                    template_output = template_output.repeat(batch_size, 1)
                else:
                    # 截取输出以匹配批次大小
                    template_output = template_output[:batch_size]

            return template_output
            
        except Exception as e:
            print(f"Retrieval_Retro.forward出错: {e}")
            # 如果可能，尝试返回一个有效的输出
            num_samples = 1
            if isinstance(data, tuple) and len(data) > 0 and hasattr(data[0], 'ptr'):
                num_samples = len(data[0].ptr) - 1
            # 返回全零输出
            return torch.zeros((num_samples, self.output_dim), device=self.device)    
