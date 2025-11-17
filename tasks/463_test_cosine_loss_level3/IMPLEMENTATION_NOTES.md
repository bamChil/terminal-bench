# Terminal-Bench LV3任务实现说明

## 任务概述
- **任务名称**: Liger-Kernel cosine loss (LV3)
- **任务级别**: Level 3（完整仓库任务）
- **仓库名称**: Liger-Kernel
- **基础镜像**: pb-python310_cu121_torch251-base_08bcbe5c
- **实例镜像**: pb-instance_ffbb2d38 (预安装环境)

## 实现参考
参考 `run_infer_for_dev.py` 中的逻辑：
- `initialize_runtime()` 第689-760行 - LV3初始化逻辑 → Dockerfile
- `complete_runtime()` - 评测逻辑 → run-tests.sh

---

## Level 3 与 Level 1 的关键区别

### LV1 任务特点
- **代码掩码任务**：给定部分完整代码，agent需要补全缺失部分
- 执行 `replace_with_masked_code.py` 脚本进行代码掩码
- `/testbed` 从 `/root/my_repo1` 复制完整仓库，然后用掩码代码替换
- Agent的任务是"填空"

### LV3 任务特点
- **完整仓库任务**：agent需要从头构建完整的代码库
- **不执行** `replace_with_masked_code.py` 脚本
- `/testbed` 初始为空（或只有基础结构）
- Agent的任务是"从零构建"

---

## 1. Dockerfile 修改说明

### 实现的LV3初始化逻辑（参考initialize_runtime第689-760行）

根据 `run_infer_for_dev.py` 中的 `initialize_runtime` 函数，LV3任务在预安装环境下需要执行以下操作：

#### 关键代码对应：

```python
# 第696-697行：进行仓库预处理（在宿主机操作，terminal-bench中不需要）
preprocess_lv3_repository(runtime, test_host_path, repo_name, logger)

# 第699-706行：删除/testbed里所有内容
init_testbed_cmd = 'rm -rf /testbed/*'
```

**重要区别**：LV3 **不从** `/root/my_repo1` **复制代码**，而LV1会复制。

#### Dockerfile实现步骤：

1. **清空 /testbed**
   ```dockerfile
   RUN rm -rf /testbed/*
   ```
   - 删除 /testbed 中所有内容
   - **不从** /root/my_repo1 复制（这是与LV1的关键区别）
   - Agent将在空白环境中从头构建代码

2. **清理 /root 目录**
   ```dockerfile
   RUN find /root -maxdepth 1 ! -name ".*" ! -path "/root" -exec rm -rf {} +
   ```
   - 删除 /root 下所有非隐藏文件和文件夹
   - 保留隐藏文件（如 .bashrc 等）

3. **复制task目录**
   ```dockerfile
   COPY task /tmp/task
   ```
   - 复制 task 目录到临时位置
   - **注意**：LV3不执行 replace_with_masked_code.py 脚本

4. **清理不需要的文件**
   ```dockerfile
   RUN rm -rf /tmp/task/notes && \
       rm -rf /tmp/task/origin && \
       rm -f /tmp/task/replace_with_masked_code.py
   ```
   - 删除 notes/ 目录
   - 删除 origin/ 目录
   - 删除 replace_with_masked_code.py 脚本（LV3不需要）

5. **移动task到工作目录**
   ```dockerfile
   RUN mkdir -p /workspace/task && \
       mv /tmp/task/* /workspace/task/ && \
       rm -rf /tmp/task
   ```
   - 将清理后的 task 目录移动到 /workspace/task
   - Agent 将从这个位置读取 prompt.md 等任务文件

6. **删除 .git 文件夹**
   ```dockerfile
   RUN rm -rf /testbed/.git || true
   ```
   - 移除版本控制信息

### 关键点
- 使用预安装环境镜像 `pb-instance_ffbb2d38`
- `/testbed` 初始为空，让agent从头构建
- 工作目录设置为 `/testbed`
- Agent 接收任务的位置为 `/workspace/task`

---

## 2. run-tests.sh 修改说明

### 实现的评测逻辑（参考complete_runtime）

LV3的评测流程与LV1类似，主要区别在于任务性质（完整构建 vs 代码补全）：

1. **读取测试文件列表**
   - 从 `$TEST_DIR/path2test.txt` 读取所有测试文件路径
   - 格式：`Liger-Kernel/test/chunked_loss/test_cosine_loss.py`

2. **复制测试文件到正确位置**
   ```bash
   # 提取测试文件名
   test_file_name=$(basename "$test_file_path")
   
   # 计算相对路径（去掉repo_name前缀）
   relative_path="${test_file_path#$REPO_NAME/}"
   
   # 复制：$TEST_DIR/test_cosine_loss.py → /testbed/test/chunked_loss/test_cosine_loss.py
   cp "$TEST_DIR/$test_file_name" "/testbed/$relative_path"
   ```

3. **运行 pytest 测试**
   ```bash
   pytest $TEST_FILES -rA
   ```
   - 一次性运行所有测试文件
   - 使用 `-rA` 参数显示所有测试结果摘要
   - terminal-bench 会自动捕获并解析 pytest 输出

### preprocess_lv3_repository 简化说明

在 `run_infer_for_dev.py` 中，`preprocess_lv3_repository` 函数（第215-363行）执行复杂的仓库预处理：
1. 从容器 /root/my_repo1 复制到宿主机 test/repo_name
2. 处理测试文件的复杂路径关系

**在 terminal-bench 中的简化**：
- ✅ 不需要从容器复制仓库（agent的代码已经在/testbed）
- ✅ 只需要将测试文件从 $TEST_DIR 复制到 /testbed
- ✅ 运行 pytest，terminal-bench自动处理结果

---

## 3. 测试文件路径说明

### path2test.txt 格式
```
Liger-Kernel/test/chunked_loss/test_cosine_loss.py
Liger-Kernel/test/transformers/test_dyt.py
Liger-Kernel/test/transformers/test_sparsemax.py
Liger-Kernel/test/transformers/test_softmax.py
Liger-Kernel/test/transformers/test_fused_add_rms_norm.py
Liger-Kernel/test/transformers/test_rope.py
```

### 文件映射关系
| 源文件位置 | 目标位置 |
|-----------|---------|
| `$TEST_DIR/test_cosine_loss.py` | `/testbed/test/chunked_loss/test_cosine_loss.py` |
| `$TEST_DIR/test_dyt.py` | `/testbed/test/transformers/test_dyt.py` |
| `$TEST_DIR/test_sparsemax.py` | `/testbed/test/transformers/test_sparsemax.py` |
| ... | ... |

---

## 4. 与 run_infer_for_dev.py 的对应关系

### Dockerfile ← initialize_runtime() LV3部分

| run_infer_for_dev.py (第689-760行) | Dockerfile |
|---------------------|------------|
| `preprocess_lv3_repository(...)` | ❌ 不需要（宿主机操作，terminal-bench简化） |
| `CmdRunAction('rm -rf /testbed/*')` | `RUN rm -rf /testbed/*` |
| `CmdRunAction('find /root -maxdepth 1 ...')` | `RUN find /root -maxdepth 1 ...` |
| `runtime.copy_to(task_host_path, sandbox_task_path)` | `COPY task /tmp/task` |
| `CmdRunAction('rm -rf /{WORK_DIR}/task/notes')` | `RUN rm -rf /tmp/task/notes` |
| `CmdRunAction('rm -rf /testbed/.git')` | `RUN rm -rf /testbed/.git` |
| **不执行** `replace_with_masked_code.py` | **不包含** 执行脚本的RUN命令 |

### run-tests.sh ← complete_runtime()

| run_infer_for_dev.py | run-tests.sh |
|---------------------|--------------|
| `run_pytest_and_evaluate(runtime, instance, ...)` | `pytest $TEST_FILES -rA` |
| 复制测试文件到容器 | 从 `$TEST_DIR` 复制到 `/testbed` |
| 解析 pytest 输出，保存结果 | ❌ 不需要（terminal-bench自动处理） |
| 复制结果回宿主机 | ❌ 不需要（terminal-bench自动处理） |

---

## 5. LV1 vs LV3 对比总结

| 特性 | LV1（代码掩码） | LV3（完整构建） |
|------|---------------|----------------|
| **任务类型** | 补全缺失代码 | 从零构建完整仓库 |
| **初始/testbed状态** | 包含完整仓库+掩码 | 空目录 |
| **是否复制/root/my_repo1** | ✅ 是 | ❌ 否 |
| **执行replace_with_masked_code.py** | ✅ 是 | ❌ 否 |
| **Agent任务难度** | 中等（填空） | 高（从零构建） |
| **评测逻辑** | 相同（复制测试文件→运行pytest） | 相同 |

---

## 6. 验证清单

### Dockerfile 验证
- [x] FROM 使用正确的实例镜像（pb-instance_ffbb2d38）
- [x] 删除 /testbed/*（**不复制** /root/my_repo1）
- [x] 清理 /root 目录
- [x] **不执行** replace_with_masked_code.py
- [x] 清理临时文件（notes, origin, replace_with_masked_code.py）
- [x] 删除 .git 文件夹
- [x] 移动 task 到 /workspace/task
- [x] 设置 WORKDIR 为 /testbed

### run-tests.sh 验证
- [x] 激活 conda 环境（testbed）
- [x] 读取 $TEST_DIR/path2test.txt
- [x] 提取测试文件名和相对路径
- [x] 复制测试文件到 /testbed 正确位置
- [x] 运行 pytest 测试
- [x] 使用 -rA 参数显示摘要

---

## 7. 常见问题

### Q: LV3为什么不从/root/my_repo1复制代码？
A: LV3是完整构建任务，agent需要从零开始在/testbed中构建整个代码库。如果预先复制代码，就失去了任务的意义。

### Q: preprocess_lv3_repository在哪里实现？
A: 这个函数在run_infer_for_dev.py中是在宿主机上操作的（用于准备测试环境）。在terminal-bench中，由于架构不同，这部分逻辑被简化：agent的代码已经在/testbed中，只需要在run-tests.sh中复制测试文件即可。

### Q: LV3测试文件如何处理？
A: 与LV1相同：从$TEST_DIR复制到/testbed的相应位置，然后运行pytest。

### Q: 如何调试 LV3 任务？
A: 使用 terminal-bench 的交互模式：
```bash
tb tasks interact 463_test_cosine_loss_level3
```

---

## 8. 参考资料

- `run_infer_for_dev.py`: 完整的 ProgrammerBench 评测脚本
  - 第689-760行: LV3初始化逻辑
  - 第215-363行: preprocess_lv3_repository函数（宿主机操作）
- `config.yaml`: 任务配置（task_level=3, repo_name, images 等）
- `task.yaml`: terminal-bench 任务元数据
- `path2test.txt`: 测试文件路径列表

