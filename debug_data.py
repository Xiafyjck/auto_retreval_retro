import torch
import os
from torch_geometric.data import Data, Batch
import sys

def inspect_data_structure(file_path, max_samples=5):
    """详细检查数据集的结构"""
    print(f"\n===== 检查数据文件: {file_path} =====")
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            return
            
        # 加载数据
        data = torch.load(file_path, weights_only=False)
        print(f"数据类型: {type(data)}")
        print(f"数据长度: {len(data) if hasattr(data, '__len__') else '不可计数'}")
        
        # 检查样本结构
        if hasattr(data, '__getitem__'):
            n_samples = min(max_samples, len(data))
            print(f"\n检查前 {n_samples} 个样本:")
            
            for i in range(n_samples):
                sample = data[i]
                print(f"\n样本 {i}:")
                print(f"  类型: {type(sample)}")
                
                if isinstance(sample, tuple) or isinstance(sample, list):
                    print(f"  长度: {len(sample)}")
                    
                    # 检查每个部分
                    for j, part in enumerate(sample):
                        print(f"  部分 {j}:")
                        print(f"    类型: {type(part)}")
                        
                        # 如果是PyG Data对象
                        if hasattr(part, 'x'):
                            print(f"    x类型: {type(part.x)}")
                            print(f"    x形状: {part.x.shape}")
                            if hasattr(part, 'edge_index'):
                                print(f"    edge_index类型: {type(part.edge_index)}")
                                print(f"    edge_index形状: {part.edge_index.shape}")
                            if hasattr(part, 'edge_attr'):
                                print(f"    edge_attr类型: {type(part.edge_attr)}")
                                print(f"    edge_attr形状: {part.edge_attr.shape}")
                            if hasattr(part, 'fc_weight'):
                                print(f"    fc_weight类型: {type(part.fc_weight)}")
                                print(f"    fc_weight形状: {part.fc_weight.shape if hasattr(part.fc_weight, 'shape') else '标量'}")
                        
                        # 如果是列表检查内部元素
                        elif isinstance(part, list):
                            print(f"    列表长度: {len(part)}")
                            if len(part) > 0:
                                first_elem = part[0]
                                print(f"    第一个元素类型: {type(first_elem)}")
                                
                                # 如果是PyG Data对象
                                if hasattr(first_elem, 'x'):
                                    print(f"    第一个元素x形状: {first_elem.x.shape}")
                                    if hasattr(first_elem, 'edge_index'):
                                        print(f"    第一个元素edge_index形状: {first_elem.edge_index.shape}")
                
                # 如果是PyG Data对象而不是元组/列表
                elif hasattr(sample, 'x'):
                    print(f"  x类型: {type(sample.x)}")
                    print(f"  x形状: {sample.x.shape}")
                    if hasattr(sample, 'edge_index'):
                        print(f"  edge_index类型: {type(sample.edge_index)}")
                        print(f"  edge_index形状: {sample.edge_index.shape}")
    
    except Exception as e:
        print(f"检查过程中出错: {e}")
        import traceback
        traceback.print_exc()

def test_collate_fn(file_path, batch_size=2):
    """测试collate_fn函数"""
    print(f"\n===== 测试自定义批处理函数 =====")
    try:
        # 加载数据
        data = torch.load(file_path, weights_only=False)
        
        # 定义自定义批处理函数
        def custom_collate_fn(batch):
            print("\n批处理输入:")
            print(f"批次类型: {type(batch)}")
            print(f"批次长度: {len(batch)}")
            
            if len(batch) > 0:
                print(f"第一个元素类型: {type(batch[0])}")
                if isinstance(batch[0], tuple) or isinstance(batch[0], list):
                    print(f"第一个元素长度: {len(batch[0])}")
                    
                    # 检查每个部分
                    for j, part in enumerate(batch[0]):
                        print(f"第一个元素部分 {j} 类型: {type(part)}")
                
            # 转换所有列表为元组
            new_batch = []
            for item in batch:
                if isinstance(item, list):
                    item = tuple(item)
                new_batch.append(item)
            
            # Batch main graphs
            try:
                main_graphs = [item[0] for item in new_batch]
                batched_main_graphs = Batch.from_data_list(main_graphs)
                
                # 检查第一组额外图
                first_additional_graphs = []
                for item in new_batch:
                    if isinstance(item[1], list):
                        first_additional_graphs.extend(item[1])
                    else:
                        print(f"警告: 额外图不是列表，而是 {type(item[1])}")
                
                # 确保列表非空
                if not first_additional_graphs:
                    print("警告: 第一组额外图为空，创建虚拟图")
                    dummy_graph = Data(
                        x=torch.zeros((1, main_graphs[0].x.size(1))),
                        edge_index=torch.zeros((2, 1), dtype=torch.long),
                        edge_attr=torch.zeros((1, main_graphs[0].edge_attr.size(1))),
                        fc_weight=torch.ones(1)
                    )
                    first_additional_graphs = [dummy_graph]
                
                batched_first_additional_graphs = Batch.from_data_list(first_additional_graphs)
                
                # 检查第二组额外图
                second_additional_graphs = []
                for item in new_batch:
                    if isinstance(item[2], list):
                        second_additional_graphs.extend(item[2])
                    else:
                        print(f"警告: 额外图不是列表，而是 {type(item[2])}")
                
                # 确保列表非空
                if not second_additional_graphs:
                    print("警告: 第二组额外图为空，创建虚拟图")
                    dummy_graph = Data(
                        x=torch.zeros((1, main_graphs[0].x.size(1))),
                        edge_index=torch.zeros((2, 1), dtype=torch.long),
                        edge_attr=torch.zeros((1, main_graphs[0].edge_attr.size(1))),
                        fc_weight=torch.ones(1)
                    )
                    second_additional_graphs = [dummy_graph]
                
                batched_second_additional_graphs = Batch.from_data_list(second_additional_graphs)
                
                # 打印结果
                print("\n批处理输出:")
                print(f"主图批次类型: {type(batched_main_graphs)}")
                print(f"主图批次x形状: {batched_main_graphs.x.shape}")
                print(f"第一组额外图批次类型: {type(batched_first_additional_graphs)}")
                print(f"第一组额外图批次x形状: {batched_first_additional_graphs.x.shape}")
                print(f"第二组额外图批次类型: {type(batched_second_additional_graphs)}")
                print(f"第二组额外图批次x形状: {batched_second_additional_graphs.x.shape}")
                
                # 测试返回元组
                result = (batched_main_graphs, batched_first_additional_graphs, batched_second_additional_graphs)
                print(f"返回结果类型: {type(result)}")
                return result
                
            except Exception as e:
                print(f"批处理过程出错: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # 模拟DataLoader处理批次
        from torch.utils.data import DataLoader
        loader = DataLoader(data, batch_size=batch_size, collate_fn=custom_collate_fn)
        
        # 获取第一个批次
        first_batch = next(iter(loader))
        
        # 检查批次结果
        print("\n最终批次:")
        print(f"批次类型: {type(first_batch)}")
        if isinstance(first_batch, tuple):
            print(f"批次长度: {len(first_batch)}")
            
            # 检查每个部分
            for i, part in enumerate(first_batch):
                print(f"批次部分 {i}:")
                print(f"  类型: {type(part)}")
                print(f"  属性: {dir(part)[:10]}...")
    
    except Exception as e:
        print(f"测试批处理函数时出错: {e}")
        import traceback
        traceback.print_exc()

def test_forward_function(file_path, batch_size=2):
    """测试模型的forward函数"""
    print(f"\n===== 测试模型forward函数 =====")
    try:
        # 加载数据
        data = torch.load(file_path, weights_only=False)
        
        # 定义模拟的forward函数
        def mock_forward(data_batch):
            print("\n模型输入:")
            print(f"输入类型: {type(data_batch)}")
            
            # 测试类型转换
            if isinstance(data_batch, list):
                print("将列表转换为元组")
                data_batch = tuple(data_batch)
            
            print(f"处理后类型: {type(data_batch)}")
            
            if isinstance(data_batch, tuple) and len(data_batch) == 3:
                print(f"元组长度: {len(data_batch)}")
                
                # 检查每个部分
                for i, part in enumerate(data_batch):
                    print(f"部分 {i}:")
                    print(f"  类型: {type(part)}")
                    
                    # 检查PyG对象
                    if hasattr(part, 'x'):
                        print(f"  x形状: {part.x.shape}")
                    elif i > 0:  # 检查额外图类型
                        if not hasattr(part, 'x'):
                            print(f"  警告: 部分 {i} 不是PyG对象，没有x属性")
                
                return "模拟前向传播成功"
            else:
                print(f"错误: 输入不是长度为3的元组")
                return None
        
        # 自定义批处理函数
        def debug_collate_fn(batch):
            # 确保batch是元组列表
            for i, item in enumerate(batch):
                if isinstance(item, list):
                    batch[i] = tuple(item)
            
            # Batch main graphs
            main_graphs = [item[0] for item in batch]
            batched_main_graphs = Batch.from_data_list(main_graphs)
            
            # 检查第一组额外图
            first_additional_graphs = []
            for item in batch:
                if isinstance(item[1], list):
                    first_additional_graphs.extend(item[1])
                else:
                    dummy_graph = Data(
                        x=torch.zeros((1, main_graphs[0].x.size(1))),
                        edge_index=torch.zeros((2, 1), dtype=torch.long),
                        edge_attr=torch.zeros((1, main_graphs[0].edge_attr.size(1))),
                        fc_weight=torch.ones(1)
                    )
                    first_additional_graphs.append(dummy_graph)
            
            # 确保列表非空
            if not first_additional_graphs:
                dummy_graph = Data(
                    x=torch.zeros((1, main_graphs[0].x.size(1))),
                    edge_index=torch.zeros((2, 1), dtype=torch.long),
                    edge_attr=torch.zeros((1, main_graphs[0].edge_attr.size(1))),
                    fc_weight=torch.ones(1)
                )
                first_additional_graphs = [dummy_graph]
            
            batched_first_additional_graphs = Batch.from_data_list(first_additional_graphs)
            
            # 检查第二组额外图
            second_additional_graphs = []
            for item in batch:
                if isinstance(item[2], list):
                    second_additional_graphs.extend(item[2])
                else:
                    dummy_graph = Data(
                        x=torch.zeros((1, main_graphs[0].x.size(1))),
                        edge_index=torch.zeros((2, 1), dtype=torch.long),
                        edge_attr=torch.zeros((1, main_graphs[0].edge_attr.size(1))),
                        fc_weight=torch.ones(1)
                    )
                    second_additional_graphs.append(dummy_graph)
            
            # 确保列表非空
            if not second_additional_graphs:
                dummy_graph = Data(
                    x=torch.zeros((1, main_graphs[0].x.size(1))),
                    edge_index=torch.zeros((2, 1), dtype=torch.long),
                    edge_attr=torch.zeros((1, main_graphs[0].edge_attr.size(1))),
                    fc_weight=torch.ones(1)
                )
                second_additional_graphs = [dummy_graph]
            
            batched_second_additional_graphs = Batch.from_data_list(second_additional_graphs)
            
            # 明确返回元组
            return (batched_main_graphs, batched_first_additional_graphs, batched_second_additional_graphs)
        
        # 模拟DataLoader处理批次
        from torch.utils.data import DataLoader
        loader = DataLoader(data, batch_size=batch_size, collate_fn=debug_collate_fn)
        
        # 获取第一个批次
        try:
            first_batch = next(iter(loader))
            
            # 测试forward函数
            result = mock_forward(first_batch)
            print(f"\n前向传播结果: {result}")
        except Exception as e:
            print(f"获取或处理批次时出错: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"测试forward函数时出错: {e}")
        import traceback
        traceback.print_exc()

def fix_data_file(file_path, output_path=None):
    """尝试修复数据文件中的问题"""
    print(f"\n===== 尝试修复数据文件: {file_path} =====")
    try:
        # 如果没有指定输出路径，则覆盖原文件
        if output_path is None:
            output_path = file_path + ".fixed"
        
        # 加载数据
        data = torch.load(file_path, weights_only=False)
        print(f"原始数据类型: {type(data)}")
        print(f"原始数据长度: {len(data) if hasattr(data, '__len__') else '不可计数'}")
        
        # 修复数据
        fixed_data = []
        for i, sample in enumerate(data):
            if isinstance(sample, list):
                # 将列表转换为元组
                sample = tuple(sample)
                print(f"样本 {i}: 列表已转换为元组")
            
            # 确保样本是元组且长度为3
            if not isinstance(sample, tuple) or len(sample) != 3:
                print(f"样本 {i}: 不是长度为3的元组，已跳过")
                continue
            
            # 确保额外图是列表
            main_graph = sample[0]
            additional_graphs_1 = sample[1] if isinstance(sample[1], list) else []
            additional_graphs_2 = sample[2] if isinstance(sample[2], list) else []
            
            # 验证主图是PyG Data对象
            if not hasattr(main_graph, 'x') or not hasattr(main_graph, 'edge_index'):
                print(f"样本 {i}: 主图不是有效的PyG Data对象，已跳过")
                continue
            
            # 检查额外图
            valid_additional_1 = []
            for j, graph in enumerate(additional_graphs_1):
                if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
                    valid_additional_1.append(graph)
                else:
                    print(f"样本 {i}, 额外图1 {j}: 不是有效的PyG Data对象，已跳过")
            
            valid_additional_2 = []
            for j, graph in enumerate(additional_graphs_2):
                if hasattr(graph, 'x') and hasattr(graph, 'edge_index'):
                    valid_additional_2.append(graph)
                else:
                    print(f"样本 {i}, 额外图2 {j}: 不是有效的PyG Data对象，已跳过")
            
            # 创建修复后的样本
            fixed_sample = (main_graph, valid_additional_1, valid_additional_2)
            fixed_data.append(fixed_sample)
        
        # 保存修复后的数据
        torch.save(fixed_data, output_path)
        print(f"修复后的数据已保存到: {output_path}")
        print(f"修复后的数据大小: {len(fixed_data)}")
        
        return fixed_data
    
    except Exception as e:
        print(f"修复数据文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# 主函数
def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python debug_data.py <数据文件路径> [--fix] [--test-collate] [--test-forward]")
        sys.exit(1)
    
    # 获取文件路径
    file_path = sys.argv[1]
    
    # 检查操作参数
    fix_file = "--fix" in sys.argv
    test_collate = "--test-collate" in sys.argv
    test_forward = "--test-forward" in sys.argv
    
    # 如果没有指定操作，则默认检查数据结构
    if not (fix_file or test_collate or test_forward):
        inspect_data_structure(file_path)
    
    # 执行请求的操作
    if fix_file:
        fixed_data = fix_data_file(file_path)
    
    if test_collate:
        test_collate_fn(file_path)
    
    if test_forward:
        test_forward_function(file_path)

if __name__ == "__main__":
    main()