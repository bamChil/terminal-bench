#!/bin/bash

# Terminal-Bench评测脚本 - LV1任务（Liger-Kernel monkey patch）
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
echo "LV1任务评测：将测试文件复制到/testbed并运行pytest"
echo "========================================="

# 读取path2test.txt文件，获取所有测试文件路径
PATH2TEST_FILE="$TEST_DIR/path2test.txt"

if [ ! -f "$PATH2TEST_FILE" ]; then
    echo "错误: path2test.txt文件不存在: $PATH2TEST_FILE"
    exit 1
fi

# repo名称（从config.yaml读取）
REPO_NAME="xarray"

echo "步骤1: 复制测试文件到/testbed..."
# 遍历path2test.txt中的每一行，复制测试文件
while IFS= read -r test_file_path; do
    # 跳过空行
    [ -z "$test_file_path" ] && continue
    
    # 提取测试文件名（例如：test_monkey_patch.py）
    test_file_name=$(basename "$test_file_path")
    
    # 计算相对于repo_name的路径（例如：test/transformers/test_monkey_patch.py）
    # test_file_path格式是：Liger-Kernel/test/transformers/test_monkey_patch.py
    relative_path="${test_file_path#$REPO_NAME/}"
    
    # 源文件路径（从$TEST_DIR获取）
    source_file="$TEST_DIR/$test_file_name"
    
    # 目标文件路径（在/testbed下）
    target_file="/testbed/$relative_path"
    
    # 检查源文件是否存在
    if [ ! -f "$source_file" ]; then
        echo "警告: 源测试文件不存在: $source_file，跳过"
        continue
    fi
    
    # 创建目标目录（如果不存在）
    target_dir=$(dirname "$target_file")
    mkdir -p "$target_dir"
    
    # 复制测试文件到目标位置
    echo "  复制: $test_file_name -> $relative_path"
    cp "$source_file" "$target_file"
    
done < "$PATH2TEST_FILE"

echo ""
echo "步骤2: 执行wrap_imports_with_try.py脚本..."
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
echo "步骤3: 运行pytest测试..."
echo "========================================="

# ===== 方式1: 只运行第一个测试文件 (当前使用) =====
# 读取path2test.txt的第一行
FIRST_TEST_PATH=$(head -n 1 "$PATH2TEST_FILE" | tr -d '[:space:]')

if [ -z "$FIRST_TEST_PATH" ]; then
    echo "错误: path2test.txt为空或第一行为空"
    exit 1
fi

# 计算相对路径和目标文件
relative_path="${FIRST_TEST_PATH#$REPO_NAME/}"
target_file="/testbed/$relative_path"

echo "只运行第一个测试: $relative_path"

# 检查文件是否存在并运行
if [ -f "$target_file" ]; then
    pytest "$target_file" -rA --color=no
else
    echo "错误: 测试文件不存在: $target_file"
    exit 1
fi

# ===== 方式2: 运行所有测试文件 (已注释，保留以备后用) =====
# # 收集所有需要测试的文件路径
# TEST_FILES=""
# while IFS= read -r test_file_path; do
#     [ -z "$test_file_path" ] && continue
#     
#     relative_path="${test_file_path#$REPO_NAME/}"
#     target_file="/testbed/$relative_path"
#     
#     # 只添加存在的文件
#     if [ -f "$target_file" ]; then
#         TEST_FILES="$TEST_FILES $target_file"
#     fi
# done < "$PATH2TEST_FILE"
# 
# # 一次性运行所有测试文件
# # terminal-bench会自动捕获pytest输出并解析结果
# if [ -n "$TEST_FILES" ]; then
#     pytest $TEST_FILES -rA --color=no
# else
#     echo "错误: 没有找到可测试的文件"
#     exit 1
# fi

echo "========================================="
echo "测试评测完成"
echo "========================================="