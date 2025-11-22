#!/bin/bash

# Terminal-Bench评测脚本 - LV3任务（Liger-Kernel cosine loss）
# 参考run_infer_for_dev.py中的complete_runtime逻辑

# Check if we're in a valid working directory
if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set. Please set a WORKDIR in your Dockerfile before running this script."
    exit 1
fi

# 激活conda环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate testbed

echo "========================================="
echo "LV3任务评测：将测试文件复制到/testbed并运行pytest"
echo "========================================="

# 读取path2test.txt文件，获取所有测试文件路径
PATH2TEST_FILE="$TEST_DIR/path2test.txt"

if [ ! -f "$PATH2TEST_FILE" ]; then
    echo "错误: path2test.txt文件不存在: $PATH2TEST_FILE"
    exit 1
fi

# repo名称（从config.yaml读取）
REPO_NAME="pytest"

echo "步骤1: LV3特殊处理 - 清空/testbed并复制完整仓库..."
echo "========================================="

# LV3特殊操作（参考utils.py第811-837行）：
# 1. 清空/testbed下的所有内容
echo "清空/testbed目录..."
rm -rf /testbed/*
if [ $? -eq 0 ]; then
    echo "✓ 成功清空/testbed目录"
else
    echo "⚠ 清空/testbed目录失败"
fi

rm -rf /testbed/* && \
cp /tmp/my_repo1.zip /testbed/ && \
cd /testbed && \
unzip -o -P ace_bench my_repo1.zip && \
rm -f my_repo1.zip

echo ""
echo "步骤2: 复制测试文件到/testbed..."
echo "========================================="

# 只读取path2test.txt的第一行并复制该测试文件
FIRST_LINE=$(head -n 1 "$PATH2TEST_FILE")
if [ -z "$FIRST_LINE" ]; then
    echo "错误: path2test.txt文件为空"
    exit 1
fi

test_file_path="$FIRST_LINE"
test_file_name=$(basename "$test_file_path")
relative_path="${test_file_path#$REPO_NAME/}"
source_file="$TEST_DIR/$test_file_name"
target_file="/testbed/$relative_path"

if [ ! -f "$source_file" ]; then
    echo "警告: 源测试文件不存在: $source_file，跳过"
else
    target_dir=$(dirname "$target_file")
    mkdir -p "$target_dir"
    echo "  复制: $test_file_name -> $relative_path"
    cp "$source_file" "$target_file"
fi

echo ""
echo "步骤3: 执行wrap_imports_with_try.py脚本..."
echo "========================================="

# 执行wrap_imports_with_try.py脚本（参考utils.py第871-882行）
# 这个脚本会处理/testbed下的import语句，使其更加健壮
WRAP_SCRIPT="$TEST_DIR/wrap_imports_with_try.py"
if [ -f "$WRAP_SCRIPT" ]; then
    echo "运行: python $WRAP_SCRIPT /testbed -r --no-backup"
    python "$WRAP_SCRIPT" /testbed -r --no-backup
    if [ $? -eq 0 ]; then
        echo "✓ wrap_imports_with_try.py执行成功"
    else
        echo "⚠ wrap_imports_with_try.py执行失败，继续测试"
    fi
else
    echo "⚠ wrap_imports_with_try.py脚本不存在，跳过"
fi

echo ""
echo "步骤4: 运行pytest测试..."
echo "========================================="

# 只读取并运行第一行的测试路径
FIRST_LINE=$(head -n 1 "$PATH2TEST_FILE")

if [ -z "$FIRST_LINE" ]; then
    echo "错误: path2test.txt文件为空"
    exit 1
fi

# 计算相对路径
relative_path="${FIRST_LINE#$REPO_NAME/}"
target_file="/testbed/$relative_path"

echo "运行第一行测试: $relative_path"

# 检查测试文件是否存在并运行
if [ -f "$target_file" ]; then
    pytest "$target_file" -rA --color=no
else
    echo "错误: 测试文件不存在: $target_file"
    exit 1
fi

echo "========================================="
echo "测试评测完成"
echo "========================================="