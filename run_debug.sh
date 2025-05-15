#!/bin/bash
# 设置模型文件和要分析的数据文件
MODEL_FILE="models/Retrieval_Retro.py"
MAIN_FILE="main_Retrieval_Retro.py"
TRAIN_FILE="./dataset/year/train_K_3.pt"
VALID_FILE="./dataset/year/valid_K_3.pt"
TEST_FILE="./dataset/year/test_K_3.pt"

# 检查Python脚本是否存在
if [ ! -f "debug_data.py" ]; then
    echo "错误: debug_data.py 文件不存在"
    exit 1
fi

# 检查模型和主文件
echo "=== 检查关键文件 ==="
if [ -f "$MODEL_FILE" ]; then
    echo "模型文件存在: $MODEL_FILE"
    echo "文件头10行:"
    head -n 10 "$MODEL_FILE"
else
    echo "警告: 模型文件不存在: $MODEL_FILE"
fi

if [ -f "$MAIN_FILE" ]; then
    echo "主文件存在: $MAIN_FILE"
    echo "custom_collate_fn函数:"
    grep -A 20 "def custom_collate_fn" "$MAIN_FILE"
else
    echo "警告: 主文件不存在: $MAIN_FILE"
fi

# 检查数据文件
echo -e "\n=== 检查数据文件 ==="
for DATA_FILE in "$TRAIN_FILE" "$VALID_FILE" "$TEST_FILE"; do
    if [ -f "$DATA_FILE" ]; then
        echo "数据文件存在: $DATA_FILE"
        echo "文件大小: $(ls -lh "$DATA_FILE" | awk '{print $5}')"
    else
        echo "警告: 数据文件不存在: $DATA_FILE"
    fi
done

# 运行数据分析
echo -e "\n=== 分析训练数据 ==="
python debug_data.py "$TRAIN_FILE"

echo -e "\n=== 测试批处理函数 ==="
python debug_data.py "$TRAIN_FILE" --test-collate

echo -e "\n=== 测试前向传播 ==="
python debug_data.py "$TRAIN_FILE" --test-forward

# 提供修复选项
echo -e "\n=== 修复选项 ==="
echo "如果需要修复数据，请运行:"
echo "python debug_data.py $TRAIN_FILE --fix"
echo "python debug_data.py $VALID_FILE --fix"
echo "python debug_data.py $TEST_FILE --fix"