#!/bin/bash

# 设置日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/preprocess_${TIMESTAMP}.log"

# 参数设置
SPLIT="year"
K=3
DEVICE=0

echo "===== 开始数据预处理 =====" | tee -a "$LOG_FILE"
echo "时间: $(date)" | tee -a "$LOG_FILE"
echo "参数: 数据集=$SPLIT, K=$K, 设备=$DEVICE" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# 处理训练数据
echo "1. 处理训练数据..." | tee -a "$LOG_FILE"
python collate.py --mode train --split "$SPLIT" --K "$K" --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
TRAIN_STATUS=$?

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "训练数据处理失败! 错误代码: $TRAIN_STATUS" | tee -a "$LOG_FILE"
    exit 1
fi

# 处理验证数据
echo "2. 处理验证数据..." | tee -a "$LOG_FILE"
python collate.py --mode valid --split "$SPLIT" --K "$K" --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
VALID_STATUS=$?

if [ $VALID_STATUS -ne 0 ]; then
    echo "验证数据处理失败! 错误代码: $VALID_STATUS" | tee -a "$LOG_FILE"
    exit 1
fi

# 处理测试数据
echo "3. 处理测试数据..." | tee -a "$LOG_FILE"
python collate.py --mode test --split "$SPLIT" --K "$K" --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
TEST_STATUS=$?

if [ $TEST_STATUS -ne 0 ]; then
    echo "测试数据处理失败! 错误代码: $TEST_STATUS" | tee -a "$LOG_FILE"
    exit 1
fi

echo "数据预处理完成!" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "完成时间: $(date)" | tee -a "$LOG_FILE"
echo "===== 结束 =====" | tee -a "$LOG_FILE"

# 设置脚本执行权限
chmod +x "$0"

# 输出帮助信息
echo ""
echo "数据处理完成后，可以运行原始的模型:"
echo "python main_Retrieval_Retro.py --seed 42 --batch_size 128 --device $DEVICE --K $K --split $SPLIT" 