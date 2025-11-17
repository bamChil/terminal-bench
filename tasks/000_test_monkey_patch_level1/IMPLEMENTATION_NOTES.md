# Terminal-Bench 任务实现说明

## 任务概述
- **任务名称**: Liger-Kernel monkey patch (LV1)
- **任务级别**: Level 1
- **仓库名称**: Liger-Kernel
- **基础镜像**: pb-python310_cu121_torch251-base_08bcbe5c
- **实例镜像**: pb-instance_ffbb2d38 (预安装环境)

## 实现参考
参考 `run_infer_for_dev.py` 中的逻辑：
- `initialize_runtime()` - 初始化逻辑 → Dockerfile
- `complete_runtime()` - 评测逻辑 → run-tests.sh

---

## 1. Dockerfile 修改说明

### 实现的LV1初始化逻辑（参考initialize_runtime）

根据 `run_infer_for_dev.py` 中的 `initialize_runtime` 函数，LV1任务在预安装环境下需要执行以下操作：

1. **初始化 /testbed**
   ```dockerfile
   RUN rm -rf /testbed/* && cp -r /root/my_repo1/* /testbed/
   ```
   - 删除 /testbed 中所有内容
   - 从 /root/my_repo1/ 复制完整仓库代码

2. **清理 /root 目录**
   ```dockerfile
   RUN find /root -maxdepth 1 ! -name ".*" ! -path "/root" -exec rm -rf {} +
   ```
   - 删除 /root 下所有非隐藏文件和文件夹
   - 保留隐藏文件（如 .bashrc 等）

3. **执行代码掩码脚本**
   ```dockerfile
   COPY task /tmp/task
   RUN cd /tmp/task && python replace_with_masked_code.py /testbed
   ```
   - 复制 task 目录到临时位置
   - 运行 `replace_with_masked_code.py` 脚本替换 /testbed 下的文件
   - 该脚本会将 notes/ 目录中的掩码代码替换到仓库中对应位置

4. **清理不需要的文件**
   ```dockerfile
   RUN rm -rf /tmp/task/notes && \
       rm -rf /tmp/task/origin && \
       rm -f /tmp/task/replace_with_masked_code.py && \
       rm -rf /tmp/task/Liger-Kernel
   ```
   - 删除 notes/ 目录（掩码代码源文件）
   - 删除 origin/ 目录（原始代码参考）
   - 删除 replace_with_masked_code.py 脚本
   - 删除 Liger-Kernel/ 仓库副本

5. **删除 .git 文件夹**
   ```dockerfile
   RUN rm -rf /testbed/.git
   ```
   - 移除版本控制信息

6. **移动 task 到工作目录**
   ```dockerfile
   RUN mkdir -p /workspace/task && \
       mv /tmp/task/* /workspace/task/ && \
       rm -rf /tmp/task
   ```
   - 将清理后的 task 目录移动到 /workspace/task
   - Agent 将从这个位置读取 prompt.md 等任务文件

### 关键点
- 使用预安装环境镜像 `pb-instance_ffbb2d38`（包含已安装的依赖）
- 工作目录设置为 `/testbed`（仓库代码位置）
- Agent 接收任务的位置为 `/workspace/task`

---

## 2. run-tests.sh 修改说明

### 实现的评测逻辑（参考complete_runtime）

根据 `run_infer_for_dev.py` 中的 `complete_runtime` 和 `run_pytest_and_evaluate` 函数，评测流程需要：

1. **读取测试文件列表**
   - 从 `$TEST_DIR/path2test.txt` 读取所有测试文件路径
   - 格式：`Liger-Kernel/test/transformers/test_monkey_patch.py`

2. **复制测试文件到正确位置**
   ```bash
   # 提取测试文件名
   test_file_name=$(basename "$test_file_path")
   
   # 计算相对路径（去掉repo_name前缀）
   relative_path="${test_file_path#$REPO_NAME/}"
   
   # 复制：$TEST_DIR/test_monkey_patch.py → /testbed/test/transformers/test_monkey_patch.py
   cp "$TEST_DIR/$test_file_name" "/testbed/$relative_path"
   ```

3. **运行 pytest 测试**
   ```bash
   pytest $TEST_FILES -rA
   ```
   - 一次性运行所有测试文件
   - 使用 `-rA` 参数显示所有测试结果摘要
   - terminal-bench 会自动捕获并解析 pytest 输出

### 简化说明
与 `run_infer_for_dev.py` 相比，terminal-bench 的评测更简洁：
- ✅ **不需要**: 复杂的 `eval_code.py` 逻辑
- ✅ **不需要**: 手动解析测试结果
- ✅ **不需要**: 将结果复制回宿主机
- ✅ **只需要**: 将测试文件移动到正确位置，然后运行 pytest

---

## 3. 测试文件路径说明

### path2test.txt 格式
```
Liger-Kernel/test/transformers/test_monkey_patch.py
Liger-Kernel/test/chunked_loss/test_cosine_loss.py
Liger-Kernel/test/chunked_loss/test_dpo_loss.py
Liger-Kernel/test/chunked_loss/test_orpo_loss.py
Liger-Kernel/test/chunked_loss/test_cpo_loss.py
Liger-Kernel/test/triton/test_triton_monkey_patch.py
```

### 文件映射关系
| 源文件位置 | 目标位置 |
|-----------|---------|
| `$TEST_DIR/test_monkey_patch.py` | `/testbed/test/transformers/test_monkey_patch.py` |
| `$TEST_DIR/test_cosine_loss.py` | `/testbed/test/chunked_loss/test_cosine_loss.py` |
| `$TEST_DIR/test_dpo_loss.py` | `/testbed/test/chunked_loss/test_dpo_loss.py` |
| ... | ... |

---

## 4. 与 run_infer_for_dev.py 的对应关系

### Dockerfile ← initialize_runtime()
| run_infer_for_dev.py | Dockerfile |
|---------------------|------------|
| `runtime.copy_to(task_host_path, sandbox_task_path)` | `COPY task /tmp/task` |
| `CmdRunAction('rm -rf /testbed/* && cp -r /root/my_repo1/* /testbed/')` | `RUN rm -rf /testbed/* && cp -r /root/my_repo1/* /testbed/` |
| `CmdRunAction('python replace_with_masked_code.py /testbed')` | `RUN cd /tmp/task && python replace_with_masked_code.py /testbed` |
| `CmdRunAction('rm -rf /{WORK_DIR}/task/notes')` | `RUN rm -rf /tmp/task/notes` |
| ... | ... |

### run-tests.sh ← complete_runtime()
| run_infer_for_dev.py | run-tests.sh |
|---------------------|--------------|
| `run_pytest_and_evaluate(runtime, instance, ...)` | `pytest $TEST_FILES -rA` |
| 复制测试文件到容器 | 从 `$TEST_DIR` 复制到 `/testbed` |
| 解析 pytest 输出，保存结果 | ❌ 不需要（terminal-bench自动处理） |
| 复制结果回宿主机 | ❌ 不需要（terminal-bench自动处理） |

---

## 5. LV3 任务支持

如果需要支持 LV3 任务，需要在 Dockerfile 中实现以下额外逻辑（参考 `preprocess_lv3_repository`）：

```dockerfile
# LV3: 从容器复制 /root/my_repo1 到本地，处理测试文件
# 这部分逻辑在 run_infer_for_dev.py 中是在宿主机上执行的
# 在 terminal-bench 中需要在 Dockerfile 或评测脚本中实现
```

**注意**: 当前任务是 LV1，暂未实现 LV3 逻辑。如需支持 LV3，请参考 `run_infer_for_dev.py` 中的 `preprocess_lv3_repository()` 函数。

---

## 6. 环境变量说明

- `$TEST_DIR`: terminal-bench 注入的环境变量，指向挂载的 tests/ 目录
- `$PWD`: 当前工作目录，应为 `/testbed`（在 Dockerfile 中设置 WORKDIR）
- `ANTHROPIC_BASE_URL`: API 基础URL（如果使用 Anthropic API）

---

## 7. 验证清单

### Dockerfile 验证
- [x] FROM 使用正确的实例镜像（pb-instance_ffbb2d38）
- [x] 初始化 /testbed（从 /root/my_repo1 复制）
- [x] 清理 /root 目录
- [x] 执行 replace_with_masked_code.py
- [x] 清理临时文件（notes, origin, etc.）
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

## 8. 常见问题

### Q: 为什么不在 run-tests.sh 中解析测试结果？
A: terminal-bench 有内置的 pytest parser，会自动捕获和解析 pytest 输出，无需手动处理。

### Q: 测试失败时会发生什么？
A: pytest 会正常返回非零退出码，terminal-bench 会将其标记为测试失败。脚本不应该使用 `|| true` 来隐藏失败。

### Q: 如何调试 Dockerfile？
A: 使用 terminal-bench 的交互模式：
```bash
tb tasks interact 000_test_monkey_patch_level1
```

### Q: 消融实验（白盒测试、browsing等）怎么办？
A: 当前实现不包含消融实验逻辑，只关注核心的 LV1 初始化和评测流程。

---

## 9. 参考资料

- `run_infer_for_dev.py`: 完整的 ProgrammerBench 评测脚本
- `config.yaml`: 任务配置（task_level, repo_name, images 等）
- `task.yaml`: terminal-bench 任务元数据
- `path2test.txt`: 测试文件路径列表

