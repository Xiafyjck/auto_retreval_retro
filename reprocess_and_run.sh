#!/bin/bash

# 设置日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/reprocess_${TIMESTAMP}.log"

# 变量设置
SPLIT="year"
K=3
DEVICE=0

echo "===== 开始重新处理数据与运行实验 =====" | tee -a "$LOG_FILE"
echo "时间: $(date)" | tee -a "$LOG_FILE"
echo "参数: 数据集=$SPLIT, K=$K, 设备=$DEVICE" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# 重新处理训练数据
echo "1. 处理训练数据..." | tee -a "$LOG_FILE"
python collate.py --mode train --split "$SPLIT" --K "$K" --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
TRAIN_STATUS=$?

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "训练数据处理失败! 错误代码: $TRAIN_STATUS" | tee -a "$LOG_FILE"
    exit 1
fi

# 重新处理验证数据
echo "2. 处理验证数据..." | tee -a "$LOG_FILE"
python collate.py --mode valid --split "$SPLIT" --K "$K" --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
VALID_STATUS=$?

if [ $VALID_STATUS -ne 0 ]; then
    echo "验证数据处理失败! 错误代码: $VALID_STATUS" | tee -a "$LOG_FILE"
    exit 1
fi

# 重新处理测试数据
echo "3. 处理测试数据..." | tee -a "$LOG_FILE"
python collate.py --mode test --split "$SPLIT" --K "$K" --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
TEST_STATUS=$?

if [ $TEST_STATUS -ne 0 ]; then
    echo "测试数据处理失败! 错误代码: $TEST_STATUS" | tee -a "$LOG_FILE"
    exit 1
fi

echo "数据处理完成!" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# 运行主实验
echo "4. 运行Retrieval-Retro实验..." | tee -a "$LOG_FILE"
python main_Retrieval_Retro.py --seed 42 --batch_size 128 --device "$DEVICE" --K "$K" --split "$SPLIT" 2>&1 | tee -a "$LOG_FILE"
MODEL_STATUS=$?

if [ $MODEL_STATUS -ne 0 ]; then
    echo "模型运行失败! 错误代码: $MODEL_STATUS" | tee -a "$LOG_FILE"
    exit 1
else
    echo "模型运行成功!" | tee -a "$LOG_FILE"
fi

echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "实验完成! 结束时间: $(date)" | tee -a "$LOG_FILE"
echo "===== 结束 =====" | tee -a "$LOG_FILE"

# 设置脚本执行权限
chmod +x "$0" 