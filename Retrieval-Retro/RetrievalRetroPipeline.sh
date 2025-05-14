#!/bin/bash

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
    # 重要: 确保这里的hidden_dim值(默认256)与main_Retrieval_Retro.py中使用的值一致
    # 否则可能导致维度不匹配错误
    python train_mpc.py --device "${GPU_ID}" --lr 0.0005 --batch_size 32 --loss adaptive --split "${DATASET_NAME}" --epochs 1000 --hidden 256
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
    # 重要: 确保这里的hidden_dim值(默认64)与之前步骤中使用的值一致
    # 可以根据需要调整，但建议在train_mpc.py和main_Retrieval_Retro.py中使用相同的hidden_dim值
    python main_Retrieval_Retro.py --device ${GPU_ID} --K ${K_VALUE} --batch_size 32 --hidden_dim 256 --epochs 1000 --lr 0.0005 --es 30 --split ${DATASET_NAME}
fi

# 返回原目录
cd "$ORIGINAL_DIR" 