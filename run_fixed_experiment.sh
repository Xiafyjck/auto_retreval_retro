#!/bin/bash

# 设置变量
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/fixed_experiment_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 输出实验信息
echo "=======================================" | tee -a "$LOG_FILE"
echo "Retrieval-Retro 接口修复实验" | tee -a "$LOG_FILE"
echo "时间: $(date)" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"

# 设置参数 - 可以根据需要修改
SEED=42
BATCH_SIZE=128
DEVICE=0
K=3
SPLIT="year"

echo "参数设置:" | tee -a "$LOG_FILE"
echo "- 随机种子: $SEED" | tee -a "$LOG_FILE"
echo "- 批次大小: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "- GPU设备: $DEVICE" | tee -a "$LOG_FILE"
echo "- 检索数量K: $K" | tee -a "$LOG_FILE"
echo "- 数据集分割: $SPLIT" | tee -a "$LOG_FILE"
echo "---------------------------------------" | tee -a "$LOG_FILE"

# 运行修复后的实验
echo "开始运行实验..." | tee -a "$LOG_FILE"
python main_Retrieval_Retro.py \
  --seed $SEED \
  --batch_size $BATCH_SIZE \
  --device $DEVICE \
  --K $K \
  --split $SPLIT \
  2>&1 | tee -a "$LOG_FILE"

# 记录退出状态
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "实验成功完成!" | tee -a "$LOG_FILE"
else
  echo "实验失败，退出代码: $EXIT_CODE" | tee -a "$LOG_FILE"
fi

echo "=======================================" | tee -a "$LOG_FILE"
echo "实验结束时间: $(date)" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE 