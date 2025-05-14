# Retrieval-Retro 复现报告

Retrieval-Retro是一种先进的无机材料逆合成推理框架，结合了材料检索和神经网络推理。本项目复现了原始论文中的模型和数据处理流程，并针对实际应用中的问题进行了优化。

## 环境搭建

Retrieval_Retro 原始代码采用了8卡A100硬编码
复现所用的环境
- [x] 8卡A100，CUDA 12.3
- [x] 单卡4090，CUDA 12.8 

依赖环境：选择了CUDA 12.6与PyTorch、PyG最新版本

```bash
conda create -n retro python=3.10 -y
conda activate retro

# core dependencies for repo
pip3 install torch

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

pip3 install -U scikit-learn

pip install tensorboardX


# packages for data processing
pip install pandas
pip install pymatgen
```

## 使用方法

整个数据处理和模型训练流程可以通过执行以下脚本完成：

```bash
./RetrievalRetroPipeline.sh <数据集名称>

# 例如：
./RetrievalRetroPipeline.sh ceder
```

该脚本会自动执行以下步骤：
1. 从原始数据生成前驱体图数据
2. 生成MPC训练/验证/测试数据集
3. 训练MPC模型用于材料检索
4. 使用MPC模型进行检索，生成候选集
5. 使用预训练NRE模型计算形成能并进行NRE检索
6. 整合MPC和NRE检索结果
7. 训练主模型Retrieval-Retro

## 复现中的Bug修复

在复现过程中，我们遇到并修复了以下关键问题：

### 1. 模型维度不匹配问题

**问题描述**：在运行主模型时遇到矩阵形状不匹配错误 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (96x128 and 512x256)`。这是由于在数据处理和模型训练过程中使用了不同的hidden_dim值导致的。

**解决方案**：
- 修改了`models/Retrieval_Retro.py`中的`fusion_linear`和`fusion_linear_2`层，使其使用动态的`hidden_dim*2`作为输入维度，而不是硬编码的512
- 在数据处理管道中确保一致使用相同的hidden_dim值（256）
- 在相关脚本调用中添加明确的参数指定：`--hidden 256`和`--hidden_dim 256`

### 2. 内存溢出问题

**问题描述**：在计算形成能差异矩阵时，由于数据集规模较大，导致GPU内存不足。

**解决方案**：
- 在`calculate_gibbs.py`中实现了分批处理大型矩阵的功能
- 将部分计算移至CPU执行，只在必要时将数据移动到GPU
- 添加了内存使用估计和自适应批处理大小功能

**问题描述**：pytorch 2.6 以上需要在 torch.load调用中加入 weights_only=False


