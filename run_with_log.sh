#!/bin/bash

# 创建日志目录
mkdir -p logs

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/experiment_${TIMESTAMP}.log"

# 运行实验并保存日志
echo "开始实验，日志保存到: $LOG_FILE"
python main_Retrieval_Retro.py --seed 42 --batch_size 128 --device 0 --K 3 --split year 2>&1 | tee $LOG_FILE

echo "实验完成!" 