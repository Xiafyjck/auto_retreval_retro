#!/bin/bash
# =============================================================================
# Retrieval-Retro 无机逆合成推理完整处理管道
# 用途：从原始数据生成到模型训练的全流程自动化脚本
# 
# 使用方法：
# ./RetrievalRetroPipeline.sh <数据集名称> [数据处理文件夹路径] [Retrieval-Retro仓库路径] [GPU_ID] [K_VALUE]
#
# 参数说明：
# <数据集名称>：要处理的数据集名称，如：ceder
# [数据处理文件夹路径]：可选，包含原始数据和处理脚本的目录，默认为./data_processing
# [Retrieval-Retro仓库路径]：可选，Retrieval-Retro代码库的目录，默认为./Retrieval-Retro
# [GPU_ID]：可选，指定使用的GPU ID，默认为0
# [K_VALUE]：可选，检索的材料数量，默认为3
#
# 示例：
# ./RetrievalRetroPipeline.sh ceder                            # 使用默认路径
# ./RetrievalRetroPipeline.sh ceder ./my_data                  # 自定义数据处理文件夹路径
# ./RetrievalRetroPipeline.sh ceder ./my_data ./my_retro 0 3   # 自定义所有参数
#
# 数据流流程说明：
# 1. 从原始数据生成前驱体图数据 (${DATASET_NAME}_precursor_graph.pt)
# 2. 从原始数据生成MPC训练/验证/测试数据集 (${DATASET_NAME}_train/val/test_mpc.pt)
# 3. 复制数据到Retrieval-Retro目录
# 4. 训练MPC模型用于检索
# 5. 使用MPC模型进行检索，生成候选集
# 6. 使用预训练NRE模型计算形成能并进行NRE检索，生成另一个候选集
# 7. 整合MPC和NRE检索结果
# 8. 生成最终训练文件 (${DATASET_NAME}/train/valid/test_K_${K_VALUE}.pt)
# 9. 运行主模型训练
# =============================================================================

# 设置错误时退出
set -e

# 获取命令行参数
if [ "$#" -lt 1 ]; then
    echo "错误: 缺少必要参数"
    echo "用法: $0 <数据集名称> [数据处理文件夹路径] [Retrieval-Retro仓库路径] [GPU_ID] [K_VALUE]"
    echo "示例: $0 ceder [./data_processing] [./Retrieval-Retro] [0] [3]"
    exit 1
fi

# 设置参数，现在数据集名称是第一个参数
DATASET_NAME="$1"

# 设置默认路径为当前工作目录下的对应文件夹
DEFAULT_PROCEED_DIR="./data_processing"
DEFAULT_RETRO_DIR="./Retrieval-Retro"

# 检查是否提供了自定义路径
if [ "$#" -ge 3 ]; then
    # 如果提供了至少3个参数，使用第2和第3个参数作为路径
    PROCEED_DIR="$2"
    RETRO_DIR="$3"
    # 可选参数，设置默认值
    GPU_ID="${4:-0}"
    K_VALUE="${5:-3}"
elif [ "$#" -ge 2 ]; then
    # 如果只提供了2个参数，第2个参数是PROCEED_DIR，RETRO_DIR使用默认值
    PROCEED_DIR="$2"
    RETRO_DIR="$DEFAULT_RETRO_DIR"
    # 可选参数，设置默认值
    GPU_ID="${3:-0}"
    K_VALUE="${4:-3}"
else
    # 如果只提供了1个参数，使用默认路径
    PROCEED_DIR="$DEFAULT_PROCEED_DIR"
    RETRO_DIR="$DEFAULT_RETRO_DIR"
    # 可选参数，设置默认值
    GPU_ID="${2:-0}"
    K_VALUE="${3:-3}"
fi

echo "=== 配置信息 ==="
echo "数据集名称: $DATASET_NAME"
echo "数据处理文件夹: $PROCEED_DIR"
echo "Retrieval-Retro仓库: $RETRO_DIR"
echo "GPU ID: $GPU_ID"
echo "K值: $K_VALUE"

# 验证必要文件是否存在
echo -e "\n=== 验证必要文件 ==="

# 检查脚本文件
SCRIPT_FILES=(
    "${PROCEED_DIR}/generate_precursor_graph_pt.py"
    "${PROCEED_DIR}/generate_mpc_split_data.py"
    "${RETRO_DIR}/train_mpc.py"
    "${RETRO_DIR}/retrieval_mpc.py"
    "${RETRO_DIR}/calculate_gibbs.py"
    "${RETRO_DIR}/collate.py"
    "${RETRO_DIR}/main_Retrieval_Retro.py"
)

for file in "${SCRIPT_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "错误: 必要的脚本文件 '$file' 不存在"
        exit 1
    else
        echo "✓ 找到脚本: $(basename "$file")"
    fi
done

# 检查原始数据文件
DATA_FILES=(
    "${PROCEED_DIR}/raw/${DATASET_NAME}_split.csv"
    "${PROCEED_DIR}/raw/${DATASET_NAME}_precursor_id.json"
    "${PROCEED_DIR}/raw/matscholar.json"
)

for file in "${DATA_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "错误: 必要的数据文件 '$file' 不存在"
        exit 1
    else
        echo "✓ 找到数据: $(basename "$file")"
    fi
done

# 检查预训练模型
PRETRAIN_MODEL="TL_pretrain(formation_exp)_embedder(graphnetwork)_lr(0.0005)_batch_size(256)_hidden(256)_seed(0)_.pt"
PRETRAIN_PATH_PROCEED="${PROCEED_DIR}/${PRETRAIN_MODEL}"
PRETRAIN_PATH_RETRO="${RETRO_DIR}/dataset/${PRETRAIN_MODEL}"

if [ -f "$PRETRAIN_PATH_RETRO" ]; then
    echo "✓ 找到预训练模型: ${PRETRAIN_MODEL}（在Retrieval-Retro目录）"
elif [ -f "$PRETRAIN_PATH_PROCEED" ]; then
    echo "✓ 找到预训练模型: ${PRETRAIN_MODEL}（在Proceed目录，将复制到Retrieval-Retro目录）"
    mkdir -p "${RETRO_DIR}/dataset"
    cp "$PRETRAIN_PATH_PROCEED" "$PRETRAIN_PATH_RETRO"
else
    echo "错误: 必要的预训练模型 '${PRETRAIN_MODEL}' 不存在"
    exit 1
fi

echo "所有必要文件检查通过"

# 验证Python环境
echo -e "\n=== 验证Python环境 ==="

# 直接使用Python执行验证代码
python -c "
import sys
required_packages = {
    'torch': 'PyTorch',
    'torch_geometric': 'PyTorch Geometric (PyG)',
    'pymatgen': 'Pymatgen',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'tqdm': 'TQDM',
    'sklearn': 'Scikit-learn'
}

missing_packages = []

for package, name in required_packages.items():
    try:
        __import__(package)
        print(f'✓ {name} 已安装')
    except ImportError:
        missing_packages.append(name)
        print(f'✗ {name} 未安装')

if missing_packages:
    print('\n错误: 缺少以下依赖包:')
    for pkg in missing_packages:
        print(f'  - {pkg}')
    sys.exit(1)
else:
    print('\n所有依赖包已安装')
    
# 检查CUDA可用性
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA可用 (设备数: {torch.cuda.device_count()})')
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('✗ CUDA不可用')
    sys.exit(1)

# 检查PyG是否正确安装
import torch_geometric
print(f'✓ PyTorch Geometric版本: {torch_geometric.__version__}')

# 尝试创建一个简单的图数据
try:
    from torch_geometric.data import Data
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.tensor([[1], [2]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    print('✓ PyG数据结构测试通过')
except Exception as e:
    print(f'✗ PyG数据结构测试失败: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "环境验证失败，请安装缺少的依赖包"
    exit 1
fi

echo "环境验证通过"

# 创建必要的目录
mkdir -p "$RETRO_DIR/dataset/${DATASET_NAME}"
# 确保proceed目录存在
mkdir -p "$PROCEED_DIR/proceed"

# 开始数据处理流程
echo -e "\n=== 开始数据处理流程 ==="

# 1. 生成前驱体图数据
echo -e "\n步骤1: 生成前驱体图数据"
PRECURSOR_GRAPH_FILE="$PROCEED_DIR/proceed/${DATASET_NAME}_precursor_graph.pt"
if [ -f "$PRECURSOR_GRAPH_FILE" ]; then
    echo "✓ 已存在前驱体图数据: $PRECURSOR_GRAPH_FILE，跳过生成步骤"
else
    # 保存当前目录
    ORIGINAL_DIR=$(pwd)
    cd "$PROCEED_DIR"
    python generate_precursor_graph_pt.py "$DATASET_NAME"
    if [ $? -ne 0 ]; then
        echo "生成前驱体图数据失败"
        # 返回原目录
        cd "$ORIGINAL_DIR"
        exit 1
    fi
    # 返回原目录
    cd "$ORIGINAL_DIR"
fi

# 2. 生成MPC训练/验证/测试数据集
echo -e "\n步骤2: 生成MPC训练/验证/测试数据集"
MPC_TRAIN_FILE="$PROCEED_DIR/proceed/${DATASET_NAME}_train_mpc.pt"
MPC_VAL_FILE="$PROCEED_DIR/proceed/${DATASET_NAME}_val_mpc.pt"
MPC_TEST_FILE="$PROCEED_DIR/proceed/${DATASET_NAME}_test_mpc.pt"

if [ -f "$MPC_TRAIN_FILE" ] && [ -f "$MPC_VAL_FILE" ] && [ -f "$MPC_TEST_FILE" ]; then
    echo "✓ 已存在MPC数据集:"
    echo "  - $MPC_TRAIN_FILE"
    echo "  - $MPC_VAL_FILE"
    echo "  - $MPC_TEST_FILE"
    echo "  跳过生成步骤"
else
    # 保存当前目录
    ORIGINAL_DIR=$(pwd)
    cd "$PROCEED_DIR"
    python generate_mpc_split_data.py "$DATASET_NAME"
    if [ $? -ne 0 ]; then
        echo "生成MPC数据集失败"
        # 返回原目录
        cd "$ORIGINAL_DIR"
        exit 1
    fi
    # 返回原目录
    cd "$ORIGINAL_DIR"
fi

# 3. 将生成的文件复制到Retrieval-Retro目录
echo -e "\n步骤3: 将生成的文件复制到Retrieval-Retro目录"
mkdir -p "$RETRO_DIR/dataset/${DATASET_NAME}"
cp "$PROCEED_DIR/proceed/${DATASET_NAME}_precursor_graph.pt" "$RETRO_DIR/dataset/${DATASET_NAME}/precursor_graph.pt"
cp "$PROCEED_DIR/proceed/${DATASET_NAME}_train_mpc.pt" "$RETRO_DIR/dataset/${DATASET_NAME}/train_mpc.pt"
cp "$PROCEED_DIR/proceed/${DATASET_NAME}_val_mpc.pt" "$RETRO_DIR/dataset/${DATASET_NAME}/valid_mpc.pt"
cp "$PROCEED_DIR/proceed/${DATASET_NAME}_test_mpc.pt" "$RETRO_DIR/dataset/${DATASET_NAME}/test_mpc.pt"
echo "✓ 已复制数据文件到Retrieval-Retro目录"

# 4. 训练MPC模型
echo -e "\n步骤4: 训练MPC模型"
# 保存当前目录
ORIGINAL_DIR=$(pwd)
cd "$RETRO_DIR"

# 检查嵌入向量文件是否存在
TRAIN_EMBEDDING="./dataset/${DATASET_NAME}/train_mpc_embeddings.pt"
VALID_EMBEDDING="./dataset/${DATASET_NAME}/valid_mpc_embeddings.pt"
TEST_EMBEDDING="./dataset/${DATASET_NAME}/test_mpc_embeddings.pt"

if [ -f "$TRAIN_EMBEDDING" ] && [ -f "$VALID_EMBEDDING" ] && [ -f "$TEST_EMBEDDING" ]; then
    echo "✓ 嵌入向量文件已存在，跳过MPC模型训练步骤"
else
    echo "嵌入向量文件不存在，开始训练MPC模型..."
    # 训练MPC模型
    python train_mpc.py --device "${GPU_ID}" --lr 0.0005 --batch_size 32 --loss adaptive --split "${DATASET_NAME}" --epochs 30
    if [ $? -ne 0 ]; then
        echo "训练MPC模型失败"
        exit 1
    fi
    
    # 再次检查是否生成了嵌入向量文件
    if [ ! -f "$TRAIN_EMBEDDING" ] || [ ! -f "$VALID_EMBEDDING" ] || [ ! -f "$TEST_EMBEDDING" ]; then
        echo "警告: 训练完成但嵌入向量文件未生成。请检查train_mpc.py是否正确保存了嵌入向量。"
        exit 1
    fi
fi

# 在训练MPC模型部分结束时返回原目录
cd "$ORIGINAL_DIR"

# 5. 使用MPC模型进行检索
echo -e "\n步骤5: 使用MPC模型进行检索（生成第一组候选集）"
# 保存当前目录并切换
ORIGINAL_DIR=$(pwd)
cd "$RETRO_DIR"

# 检查MPC检索结果是否存在
MPC_TRAIN_RESULT="./dataset/${DATASET_NAME}/train_mpc_retrieved_${K_VALUE}"
MPC_VALID_RESULT="./dataset/${DATASET_NAME}/valid_mpc_retrieved_${K_VALUE}"
MPC_TEST_RESULT="./dataset/${DATASET_NAME}/test_mpc_retrieved_${K_VALUE}"

if [ -f "$MPC_TRAIN_RESULT" ] && [ -f "$MPC_VALID_RESULT" ] && [ -f "$MPC_TEST_RESULT" ]; then
    echo "✓ MPC检索结果已存在，跳过MPC检索步骤"
else
    echo "MPC检索结果不存在，开始进行MPC检索..."
    # 检查余弦相似度矩阵文件是否存在
    COS_SIM_TRAIN="./dataset/${DATASET_NAME}/train_mpc_cos_sim_matrix.pt"
    COS_SIM_VALID="./dataset/${DATASET_NAME}/valid_mpc_cos_sim_matrix.pt"
    COS_SIM_TEST="./dataset/${DATASET_NAME}/test_mpc_cos_sim_matrix.pt"
    
    # 如果余弦相似度矩阵已存在，但检索结果不存在，说明是在计算排名时出错
    if [ -f "$COS_SIM_TRAIN" ] && [ -f "$COS_SIM_VALID" ] && [ -f "$COS_SIM_TEST" ]; then
        echo "余弦相似度矩阵已存在，但检索结果不存在，可能是在计算排名时出错。尝试重新计算..."
    fi
    
    python retrieval_mpc.py --split "${DATASET_NAME}" --K "${K_VALUE}" --device "${GPU_ID}"
    if [ $? -ne 0 ]; then
        echo "MPC检索失败"
        exit 1
    fi
fi

# 在步骤结束时返回原目录
cd "$ORIGINAL_DIR"

# 6. 计算形成能并进行NRE检索
echo -e "\n步骤6: 计算形成能并进行NRE检索（生成第二组候选集）"
# 保存当前目录并切换
ORIGINAL_DIR=$(pwd)
cd "$RETRO_DIR"

# 检查NRE检索结果是否存在
NRE_TRAIN_RESULT="./dataset/${DATASET_NAME}/train_nre_retrieved_${K_VALUE}"
NRE_VALID_RESULT="./dataset/${DATASET_NAME}/valid_nre_retrieved_${K_VALUE}"
NRE_TEST_RESULT="./dataset/${DATASET_NAME}/test_nre_retrieved_${K_VALUE}"

if [ -f "$NRE_TRAIN_RESULT" ] && [ -f "$NRE_VALID_RESULT" ] && [ -f "$NRE_TEST_RESULT" ]; then
    echo "✓ NRE检索结果已存在，跳过NRE检索步骤"
else
    echo "NRE检索结果不存在，开始进行NRE检索..."
    
    # 检查形成能文件是否存在
    FORMATION_ENERGY_TRAIN="./dataset/${DATASET_NAME}/train_formation_energy.pt"
    FORMATION_ENERGY_VALID="./dataset/${DATASET_NAME}/valid_formation_energy.pt"
    FORMATION_ENERGY_TEST="./dataset/${DATASET_NAME}/test_formation_energy.pt"
    FORMATION_ENERGY_PRECURSOR="./dataset/${DATASET_NAME}/precursor_formation_energy.pt"
    
    # 如果形成能文件已存在，但检索结果不存在，说明是在计算差异时出错
    if [ -f "$FORMATION_ENERGY_TRAIN" ] && [ -f "$FORMATION_ENERGY_VALID" ] && [ -f "$FORMATION_ENERGY_TEST" ] && [ -f "$FORMATION_ENERGY_PRECURSOR" ]; then
        echo "形成能文件已存在，但检索结果不存在，可能是在计算差异时出错。尝试重新计算..."
    fi
    
    python calculate_gibbs.py --split "${DATASET_NAME}" --K "${K_VALUE}" --device "${GPU_ID}"
    if [ $? -ne 0 ]; then
        echo "计算形成能和NRE检索失败"
        cd "$ORIGINAL_DIR"
        exit 1
    fi
fi

# 返回原目录
cd "$ORIGINAL_DIR"

# 7. 整合MPC和NRE检索结果
echo -e "\n步骤7: 整合MPC和NRE检索结果（合并两组候选集）"
# 保存当前目录并切换
ORIGINAL_DIR=$(pwd)
cd "$RETRO_DIR"

# 检查整合结果是否存在
FINAL_TRAIN_RESULT="./dataset/${DATASET_NAME}/train_final_mpc_nre_K_${K_VALUE}.pt"
FINAL_VALID_RESULT="./dataset/${DATASET_NAME}/valid_final_mpc_nre_K_${K_VALUE}.pt"
FINAL_TEST_RESULT="./dataset/${DATASET_NAME}/test_final_mpc_nre_K_${K_VALUE}.pt"

if [ -f "$FINAL_TRAIN_RESULT" ] && [ -f "$FINAL_VALID_RESULT" ] && [ -f "$FINAL_TEST_RESULT" ]; then
    echo "✓ 整合结果已存在，跳过整合步骤"
else
    echo "整合结果不存在，开始进行整合..."
    
    # 检查MPC和NRE检索结果是否都存在
    if [ ! -f "$MPC_TRAIN_RESULT" ] || [ ! -f "$MPC_VALID_RESULT" ] || [ ! -f "$MPC_TEST_RESULT" ]; then
        echo "错误: MPC检索结果不完整，无法进行整合"
        cd "$ORIGINAL_DIR"
        exit 1
    fi
    
    if [ ! -f "$NRE_TRAIN_RESULT" ] || [ ! -f "$NRE_VALID_RESULT" ] || [ ! -f "$NRE_TEST_RESULT" ]; then
        echo "错误: NRE检索结果不完整，无法进行整合"
        cd "$ORIGINAL_DIR"
        exit 1
    fi
    
    python collate.py --mode train --split "${DATASET_NAME}" --K "${K_VALUE}" --device "${GPU_ID}"
    python collate.py --mode valid --split "${DATASET_NAME}" --K "${K_VALUE}" --device "${GPU_ID}"
    python collate.py --mode test --split "${DATASET_NAME}" --K "${K_VALUE}" --device "${GPU_ID}"
    if [ $? -ne 0 ]; then
        echo "整合检索结果失败"
        cd "$ORIGINAL_DIR"
        exit 1
    fi
fi

# 返回原目录
cd "$ORIGINAL_DIR"

# 8. 将最终文件移动到正确位置
echo -e "\n步骤8: 将最终文件移动到正确位置"
# 保存当前目录并切换
ORIGINAL_DIR=$(pwd)
cd "$RETRO_DIR"

# 检查最终训练文件是否存在
TRAIN_FILE="./dataset/${DATASET_NAME}/train_K_${K_VALUE}.pt"
VALID_FILE="./dataset/${DATASET_NAME}/valid_K_${K_VALUE}.pt"
TEST_FILE="./dataset/${DATASET_NAME}/test_K_${K_VALUE}.pt"

if [ -f "$TRAIN_FILE" ] && [ -f "$VALID_FILE" ] && [ -f "$TEST_FILE" ]; then
    echo "最终训练文件已存在，跳过文件移动步骤"
else
    echo "最终训练文件不存在，开始移动文件..."
    # 确保目录存在
    mkdir -p "./dataset/${DATASET_NAME}"
    cp "./dataset/${DATASET_NAME}/train_final_mpc_nre_K_${K_VALUE}.pt" "./dataset/${DATASET_NAME}/train_K_${K_VALUE}.pt"
    cp "./dataset/${DATASET_NAME}/valid_final_mpc_nre_K_${K_VALUE}.pt" "./dataset/${DATASET_NAME}/valid_K_${K_VALUE}.pt"
    cp "./dataset/${DATASET_NAME}/test_final_mpc_nre_K_${K_VALUE}.pt" "./dataset/${DATASET_NAME}/test_K_${K_VALUE}.pt"
fi

# 返回原目录
cd "$ORIGINAL_DIR"

# 9. 运行主模型
echo -e "\n步骤9: 运行Retrieval-Retro主模型"
# 保存当前目录并切换
ORIGINAL_DIR=$(pwd)
cd "$RETRO_DIR"

# 检查模型结果文件是否存在
RESULT_FILE="./experiments/Retrieval_Retro_32_ours_Retrieval_Retro_${DATASET_NAME}_${K_VALUE}_result.txt"

if [ -f "$RESULT_FILE" ]; then
    echo "模型结果文件已存在，跳过模型训练步骤"
else
    echo "模型结果文件不存在，开始训练主模型..."
    python main_Retrieval_Retro.py --device ${GPU_ID} --K ${K_VALUE} --batch_size 32 --hidden_dim 64 --epochs 20 --lr 0.0005 --es 30 --split ${DATASET_NAME}
fi

# 返回原目录
cd "$ORIGINAL_DIR"

echo -e "\n=== 处理完成! ==="
echo "已生成并运行完毕:"
echo "- 数据集: $DATASET_NAME"
echo "- 检索数量K: $K_VALUE"
echo "- GPU ID: $GPU_ID"

echo -e "\n=== 数据流说明 ==="
echo "1. 原始数据: ${PROCEED_DIR}/raw/${DATASET_NAME}_split.csv, ${DATASET_NAME}_precursor_id.json, matscholar.json"
echo "2. 前驱体图数据: proceed/${DATASET_NAME}_precursor_graph.pt → ${RETRO_DIR}/dataset/${DATASET_NAME}/precursor_graph.pt"
echo "3. MPC数据集: proceed/${DATASET_NAME}_train/val/test_mpc.pt → ${RETRO_DIR}/dataset/${DATASET_NAME}/train/valid/test_mpc.pt"
echo "4. MPC模型训练（用于相似材料检索）"
echo "5. MPC检索结果: ${RETRO_DIR}/dataset/${DATASET_NAME}/train_mpc_retrieved_${K_VALUE}等"
echo "6. 形成能计算: ${RETRO_DIR}/dataset/${DATASET_NAME}/precursor_formation_energy.pt等"
echo "7. NRE检索结果: ${RETRO_DIR}/dataset/${DATASET_NAME}/train_nre_retrieved_${K_VALUE}等" 
echo "8. 整合检索结果: ${RETRO_DIR}/dataset/${DATASET_NAME}/train_final_mpc_nre_K_${K_VALUE}.pt等"
echo "9. 最终训练数据: ${RETRO_DIR}/dataset/${DATASET_NAME}/train_K_${K_VALUE}.pt等"
echo "10. Retrieval-Retro模型训练与评估"