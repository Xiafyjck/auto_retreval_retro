#!/bin/bash

# 设置变量
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/experiment_${TIMESTAMP}.log"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 打印基本信息
echo "=======================================" | tee -a "$LOG_FILE"
echo "Retrieval-Retro 实验开始: $(date)" | tee -a "$LOG_FILE"
echo "保存日志到: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"

# 运行实验
echo "正在运行实验..." | tee -a "$LOG_FILE"

# 可以传入实验参数
SEED=${1:-42}  # 默认种子为42
BATCH_SIZE=${2:-128}  # 默认批次大小为128
DEVICE=${3:-0}  # 默认设备为0
K=${4:-3}  # 默认检索数量为3
SPLIT=${5:-"year"}  # 默认数据集分割为year

# 打印参数
echo "参数设置:" | tee -a "$LOG_FILE"
echo "  - 随机种子: $SEED" | tee -a "$LOG_FILE"
echo "  - 批次大小: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  - GPU设备: $DEVICE" | tee -a "$LOG_FILE"
echo "  - 检索数量K: $K" | tee -a "$LOG_FILE"
echo "  - 数据集分割: $SPLIT" | tee -a "$LOG_FILE"
echo "---------------------------------------" | tee -a "$LOG_FILE"

# 执行Python脚本并记录输出
python main_Retrieval_Retro.py \
  --seed $SEED \
  --batch_size $BATCH_SIZE \
  --device $DEVICE \
  --K $K \
  --split $SPLIT \
  2>&1 | tee -a "$LOG_FILE"

# 记录结束时间和状态
EXIT_CODE=$?
echo "---------------------------------------" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
  echo "实验成功完成! 结束时间: $(date)" | tee -a "$LOG_FILE"
else
  echo "实验执行失败! 退出代码: $EXIT_CODE. 结束时间: $(date)" | tee -a "$LOG_FILE"
fi
echo "=======================================" | tee -a "$LOG_FILE"

# 设置脚本执行权限
chmod +x "$0"

exit $EXIT_CODE 