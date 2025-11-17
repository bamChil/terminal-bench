import json
import os
import tempfile
import shutil
import pandas as pd
from typing import Dict, Any, Optional
from contextlib import nullcontext
from openhands.events.action import CmdRunAction
from openhands.events.action.files import FileReadAction
from openhands.events.action.commands import CmdRunAction

try:
    from filelock import FileLock  # 可选
except Exception:
    FileLock = None

def upsert_instance_status_jsonl(
    dir_path: str,
    record: Dict[str, Any],
    filename: str = "instance_status.json",
    id_field: str = "instance_id",
    use_lock: bool = True,
) -> str:
    assert id_field in record and record[id_field] is not None, f"record 必须包含有效的 {id_field}"
    file_path = os.path.join(dir_path, filename)
    os.makedirs(dir_path, exist_ok=True)

    lock: Optional[Any] = FileLock(file_path + ".lock") if (use_lock and FileLock is not None) else None
    ctx = lock or nullcontext()

    def _write_json_line(fp, obj):
        fp.write(json.dumps(obj, ensure_ascii=False))
        fp.write("\n")

    with ctx:
        replaced = False
        # 关键修改：在 **同目录** 下建临时文件，避免跨设备重命名
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", dir=dir_path  # <—— 这里指定 dir
        ) as tmp:
            tmp_path = tmp.name
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as src:
                    for line in src:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            tmp.write(line + "\n")  # 保留无法解析的行
                            continue
                        if isinstance(obj, dict) and obj.get(id_field) == record[id_field]:
                            _write_json_line(tmp, record)  # 替换
                            replaced = True
                        else:
                            _write_json_line(tmp, obj)
            if not replaced:
                _write_json_line(tmp, record)

        # 原子替换（同一文件系统内）
        try:
            os.replace(tmp_path, file_path)
        except OSError:
            # 极端情况下的兜底（不保证原子性，但避免失败）
            shutil.move(tmp_path, file_path)

    return file_path

def extract_instance_metrics(structured_logs_dir: str, instance_id: str, logger=None) -> dict:
    """
    从结构化日志中提取实例的 token、成本和运行时间信息。
    """
    log_file = os.path.join(structured_logs_dir, f"{instance_id}.jsonl")

    if not os.path.exists(log_file):
        return {
            'total_tokens': 0,
            'total_cost': 0.0,
            'runtime_seconds': 0.0
        }

    # 从最后一个记录获取信息
    accumulated_token_usage = None
    accumulated_cost = None
    start_time = None
    end_time = None

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找最后的accumulated_cost（系统记录的真实成本）
        import re
        cost_matches = re.findall(r'"accumulated_cost":\s*([\d.]+)', content)
        if cost_matches:
            accumulated_cost = float(cost_matches[-1])

        # 查找所有的accumulated_token_usage
        token_usage_matches = re.findall(r'"accumulated_token_usage":\s*{[^}]*}', content)
        if token_usage_matches:
            # 获取最后一个token usage信息
            last_usage_str = token_usage_matches[-1]
            # 提取完整的JSON对象
            token_usage_json = '{' + last_usage_str + '}'
            try:
                token_data = json.loads(token_usage_json)
                accumulated_token_usage = token_data.get('accumulated_token_usage')
            except:
                # 如果解析失败，尝试更精确的正则表达式
                prompt_match = re.search(r'"prompt_tokens":\s*(\d+)', last_usage_str)
                completion_match = re.search(r'"completion_tokens":\s*(\d+)', last_usage_str)
                if prompt_match and completion_match:
                    accumulated_token_usage = {
                        'prompt_tokens': int(prompt_match.group(1)),
                        'completion_tokens': int(completion_match.group(1))
                    }

        # 查找开始和结束时间戳
        time_matches = re.findall(r'"timestamp":\s*"([^"]+)"', content)
        if time_matches:
            start_time = time_matches[0]
            end_time = time_matches[-1]

    except Exception as e:
        logger.warning(f"Error parsing structured log for {instance_id}: {e}")

    # 计算总 token 数
    total_tokens = 0
    if accumulated_token_usage:
        total_tokens = (
            accumulated_token_usage.get('prompt_tokens', 0) +
            accumulated_token_usage.get('completion_tokens', 0)
        )

    # 使用系统记录的真实成本，如果没有则回退到估算
    total_cost = 0.0
    if accumulated_cost is not None:
        total_cost = accumulated_cost
    elif accumulated_token_usage:
        # 备选：使用简单估算
        prompt_tokens = accumulated_token_usage.get('prompt_tokens', 0)
        completion_tokens = accumulated_token_usage.get('completion_tokens', 0)
        total_cost = (prompt_tokens * 3 / 1_000_000) + (completion_tokens * 15 / 1_000_000)

    # 计算运行时间
    runtime_seconds = 0.0
    if start_time and end_time:
        try:
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            runtime_seconds = (end_dt - start_dt).total_seconds()
        except Exception:
            runtime_seconds = 0.0

    return {
        'total_tokens': total_tokens,
        'total_cost': total_cost,
        'runtime_seconds': runtime_seconds
    }

def detect_cheating_in_logs(structured_logs_dir: str, instance_id: str, logger=None) -> Dict[str, Any]:
    """
    检测结构化日志中的作弊行为

    Args:
        structured_logs_dir: 结构化日志目录
        instance_id: 实例ID
        logger: 日志记录器

    Returns:
        Dict包含作弊检测结果:
        {
            'is_cheating': bool,
            'cheating_details': list,  # 作弊行为详情列表
            'total_violations': int    # 违规总数
        }
    """
    log_file = os.path.join(structured_logs_dir, f"{instance_id}.jsonl")

    if not os.path.exists(log_file):
        return {
            'is_cheating': False,
            'cheating_details': [],
            'total_violations': 0
        }

    cheating_details = []

    # 定义作弊规则
    CHEATING_RULES = {
        'forbidden_system_access': [
            # r'"message".*find /usr/local/lib/python\d+\.\d+',
            # r'"command".*find /usr/local/lib/python\d+\.\d+',
            # r'running command.*find /usr/local/lib/python\d+\.\d+',
            r'"message".*cat /usr/local/lib/python\d+\.\d+',
            r'"command".*cat /usr/local/lib/python\d+\.\d+',
            r'"message".*reading file: /usr/local/lib/python\d+\.\d+',
            r'"message".*reading /usr/local/lib/python\d+\.\d+',
            # r'"message".*ls /usr/local/lib/python\d+\.\d+',
            # r'"command".*ls /usr/local/lib/python\d+\.\d+',
            # r'"path".*/usr/local/lib/python\d+\.\d+/',
        ],
    }

    try:
        # 先尝试读取整个文件内容
        with open(log_file, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # 使用正则表达式直接搜索文件内容中的违规模式
        import re
        line_number = 0

        # 按行分割文件内容进行检查
        for line_num, line in enumerate(file_content.split('\n'), 1):
            if not line.strip():
                continue

            line_content = line.lower()

            # 检查所有规则
            for rule_category, patterns in CHEATING_RULES.items():
                for pattern in patterns:
                    if re.search(pattern, line_content):
                        cheating_details.append({
                            'rule_category': rule_category,
                            'pattern': pattern,
                            'line_number': line_num,
                            'matched_content': line[:200],  # 截取前200字符避免太长
                            'violation_type': f'匹配到禁止模式: {pattern}'
                        })

        # 如果没有找到违规，也尝试JSONL解析方式作为备份
        if not cheating_details:
            with open(log_file, 'r', encoding='utf-8') as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    try:
                        log_entry = json.loads(line.strip())
                        if not isinstance(log_entry, dict):
                            continue

                        # 检查JSON内容
                        content = log_entry.get('content', '')
                        message = log_entry.get('message', '')
                        command = log_entry.get('command', '')
                        path = log_entry.get('path', '')

                        # 合并所有需要检查的文本内容
                        text_to_check = f"{content} {message} {command} {path}".lower()

                        # 检查禁止命令和系统访问
                        for rule_category, patterns in CHEATING_RULES.items():
                            for pattern in patterns:
                                if re.search(pattern, text_to_check):
                                    cheating_details.append({
                                        'rule_category': rule_category,
                                        'pattern': pattern,
                                        'line_number': line_number,
                                        'matched_content': content[:200],  # 截取前200字符避免太长
                                        'violation_type': f'匹配到禁止模式: {pattern}'
                                    })

                        # 特殊检查：文件读取操作
                        if log_entry.get('action') == 'read' or 'read_file' in text_to_check:
                            file_path = ''
                            # 尝试从不同字段提取文件路径
                            if 'file' in log_entry:
                                file_path = str(log_entry['file']).lower()
                            elif 'path' in log_entry:
                                file_path = str(log_entry['path']).lower()
                            elif 'filePath' in log_entry:
                                file_path = str(log_entry['filePath']).lower()

                            if file_path:
                                for pattern in CHEATING_RULES['forbidden_file_access']:
                                    if re.search(pattern, file_path):
                                        cheating_details.append({
                                            'rule_category': 'forbidden_file_access',
                                            'pattern': pattern,
                                            'line_number': line_number,
                                            'matched_content': file_path,
                                            'violation_type': f'尝试读取禁止的文件: {file_path}'
                                        })

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_number} in {instance_id}: {e}")
                        continue

    except Exception as e:
        logger.error(f"Error reading structured log for {instance_id}: {e}")
        return {
            'is_cheating': False,
            'cheating_details': [],
            'total_violations': 0
        }

    total_violations = len(cheating_details)
    is_cheating = total_violations > 0

    if is_cheating and logger:
        logger.warning(f"Detected {total_violations} potential cheating violations in {instance_id}")

    return {
        'is_cheating': is_cheating,
        'cheating_details': cheating_details,
        'total_violations': total_violations
    }


def generate_evaluation_summary_report(
    eval_output_dir: str,
    dataset: pd.DataFrame,
    submitted_instances_count: Optional[int] = None,
    logger=None
) -> None:
    """
    生成评估摘要报告
    Args:
        eval_output_dir: 评估输出目录路径
        dataset: 数据集DataFrame
        submitted_instances_count: 提交评估的实例数量，如果为None则使用dataset长度
        logger: 日志记录器，可选
    """
    if logger:
        logger.info('Generating evaluation summary report...')

    instance_status_path = os.path.join(eval_output_dir, 'instance_status.txt')
    repos_dir = os.path.join(eval_output_dir, 'repos')
    structured_logs_dir = os.path.join(eval_output_dir, 'structured_logs')
    summary_md_path = os.path.join(eval_output_dir, 'report.md')

    if not os.path.exists(instance_status_path):
        if logger:
            logger.warning(f"Instance status file not found at {instance_status_path}, skipping summary generation.")
        return

    # 收集所有实例的结果
    instance_results = {}
    resolved_count = 0
    cheating_count = 0

    # 1. 从 instance_status.txt 读取并分类所有处理过的实例
    with open(instance_status_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                instance_id = record.get("instance_id")
                if not instance_id:
                    continue

                # 提取 token、成本和运行时间信息
                metrics = extract_instance_metrics(structured_logs_dir, instance_id, logger)

                # 检测作弊行为
                cheating_result = detect_cheating_in_logs(structured_logs_dir, instance_id, logger)

                if record.get("status") == "success" and not cheating_result['is_cheating']:
                    repo_path = record.get("repo_json_path")
                    if repo_path and os.path.exists(repo_path):
                        with open(repo_path, 'r') as repo_f:
                            # 读取 repo.json 的内容
                            repo_content = json.load(repo_f)

                            # 解析测试结果
                            summary = repo_content.get('summary', {})
                            total_tests = summary.get('total', 0)
                            passed_tests = summary.get('passed', 0)
                            failed_tests = summary.get('failed', 0)
                            error_tests = summary.get('error', 0)

                            # 计算通过率
                            test_pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

                            # 判断是否解决（根据是否有失败或错误的测试来判断）
                            resolved = failed_tests == 0 and error_tests == 0 and total_tests > 0

                            if resolved:
                                resolved_count += 1

                            instance_results[instance_id] = {
                                'resolved': resolved,
                                'total_tests': total_tests,
                                'passed_tests': passed_tests,
                                'failed_tests': failed_tests,
                                'error_tests': error_tests,
                                'test_pass_rate': test_pass_rate,
                                'test_file': 'test suite',
                                'repo_content': repo_content,
                                'total_tokens': metrics['total_tokens'],
                                'total_cost': metrics['total_cost'],
                                'runtime_seconds': metrics['runtime_seconds'],
                                'is_cheating': False,
                                'cheating_details': []
                            }
                    else:
                        instance_results[instance_id] = {
                            'resolved': False,
                            'total_tests': 0,
                            'passed_tests': 0,
                            'failed_tests': 0,
                            'error_tests': 0,
                            'test_pass_rate': 0,
                            'test_file': 'repo.json not found',
                            'repo_content': None,
                            'total_tokens': metrics['total_tokens'],
                            'total_cost': metrics['total_cost'],
                            'runtime_seconds': metrics['runtime_seconds'],
                            'is_cheating': False,
                            'cheating_details': []
                        }
                elif cheating_result['is_cheating']:
                    # 发现作弊行为，但仍然要显示测试结果数据
                    cheating_count += 1

                    # 尝试获取测试结果数据（如果有的话）
                    repo_path = record.get("repo_json_path")
                    if repo_path and os.path.exists(repo_path):
                        try:
                            with open(repo_path, 'r') as repo_f:
                                repo_content = json.load(repo_f)

                                # 解析测试结果
                                summary = repo_content.get('summary', {})
                                total_tests = summary.get('total', 0)
                                passed_tests = summary.get('passed', 0)
                                failed_tests = summary.get('failed', 0)
                                error_tests = summary.get('error', 0)

                                # 计算通过率
                                test_pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

                                instance_results[instance_id] = {
                                    'resolved': False,  # 作弊不算解决
                                    'total_tests': total_tests,
                                    'passed_tests': passed_tests,
                                    'failed_tests': failed_tests,
                                    'error_tests': error_tests,
                                    'test_pass_rate': test_pass_rate,
                                    'test_file': 'test suite',
                                    'repo_content': repo_content,
                                    'total_tokens': metrics['total_tokens'],
                                    'total_cost': metrics['total_cost'],
                                    'runtime_seconds': metrics['runtime_seconds'],
                                    'is_cheating': True,
                                    'cheating_details': cheating_result['cheating_details']
                                }
                        except:
                            # 如果无法读取测试结果，设置为默认值
                            instance_results[instance_id] = {
                                'resolved': False,
                                'total_tests': 0,
                                'passed_tests': 0,
                                'failed_tests': 0,
                                'error_tests': 0,
                                'test_pass_rate': 0,
                                'test_file': 'repo.json not found',
                                'repo_content': None,
                                'total_tokens': metrics['total_tokens'],
                                'total_cost': metrics['total_cost'],
                                'runtime_seconds': metrics['runtime_seconds'],
                                'is_cheating': True,
                                'cheating_details': cheating_result['cheating_details']
                            }
                    else:
                        # 没有测试结果文件
                        instance_results[instance_id] = {
                            'resolved': False,
                            'total_tests': 0,
                            'passed_tests': 0,
                            'failed_tests': 0,
                            'error_tests': 0,
                            'test_pass_rate': 0,
                            'test_file': 'repo.json not found',
                            'repo_content': None,
                            'total_tokens': metrics['total_tokens'],
                            'total_cost': metrics['total_cost'],
                            'runtime_seconds': metrics['runtime_seconds'],
                            'is_cheating': True,
                            'cheating_details': cheating_result['cheating_details']
                        }
                else:
                    instance_results[instance_id] = {
                        'resolved': False,
                        'total_tests': 0,
                        'passed_tests': 0,
                        'failed_tests': 0,
                        'error_tests': 0,
                        'test_pass_rate': 0,
                        'test_file': record.get("error", "Unknown error"),
                        'repo_content': None,
                        'total_tokens': metrics['total_tokens'],
                        'total_cost': metrics['total_cost'],
                        'runtime_seconds': metrics['runtime_seconds'],
                        'is_cheating': False,
                        'cheating_details': []
                    }
            except json.JSONDecodeError:
                continue

    # 2. 计算整体统计
    total_instances = len(dataset) if submitted_instances_count is None else submitted_instances_count
    failed_count = len([v for v in instance_results.values() if not v.get('resolved', False) and not v.get('is_cheating', False)])
    resolution_rate = (resolved_count / total_instances * 100) if total_instances > 0 else 0
    failure_rate = (failed_count / total_instances * 100) if total_instances > 0 else 0
    cheating_rate = (cheating_count / total_instances * 100) if total_instances > 0 else 0

    # 3. 生成markdown报告
    from datetime import datetime

    md_content = f"""# Programmer Bench Real World 测试结果报告

报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 整体统计

- **总实例数**: {total_instances}
- **已解决实例数**: {resolved_count}
- **失败实例数**: {failed_count}
- **作弊实例数**: {cheating_count}
- **解决率**: {resolution_rate:.1f}%
- **失败率**: {failure_rate:.1f}%
- **作弊率**: {cheating_rate:.1f}%

## 详细结果

| Instance ID | 状态 | 解决 | Pass | Fail | Error | Total | Pass率 | Tokens | Cost (¥) | 时间 (s) |
|-------------|------|------|------|------|-------|-------|--------|---------|----------|----------|"""

    # 按 instance_id 排序显示结果
    all_instance_ids = [row['instance_id'] for _, row in dataset.iterrows()]
    for instance_id in sorted(all_instance_ids):
        result = instance_results.get(instance_id, {
            'resolved': False,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'test_pass_rate': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'runtime_seconds': 0.0,
            'is_cheating': False,
            'cheating_details': []
        })

        # 状态列按原来的逻辑：是否有测试数据
        status = '✅' if result.get('total_tests', 0) > 0 else '❌'

        # 解决列：根据是否作弊显示不同的标识
        if result.get('is_cheating', False):
            resolved = '🚫'  # 作弊标识
        else:
            resolved = '🟢' if result.get('resolved', False) else '🔴'

        pass_rate = f"{result.get('test_pass_rate', 0):.1f}%"
        tokens = result.get('total_tokens', 0)
        cost = f"{result.get('total_cost', 0.0):.4f}"
        runtime = f"{result.get('runtime_seconds', 0.0):.1f}"

        md_content += f"\n| {instance_id} | {status} | {resolved} | {result.get('passed_tests', 0)} | {result.get('failed_tests', 0)} | {result.get('error_tests', 0)} | {result.get('total_tests', 0)} | {pass_rate} | {tokens} | {cost} | {runtime} |"

    md_content += f"""

### 图例
- **状态**: ✅ 有测试数据 | ❌ 无测试数据
- **解决**: 🟢 已解决 (无失败无错误) | 🔴 未解决 | 🚫 作弊违规

"""

    # 添加作弊行为详情
    cheating_instances = {k: v for k, v in instance_results.items() if v.get('is_cheating', False)}

    if cheating_instances:
        md_content += """## 作弊行为详情

以下实例检测到作弊行为，已被自动标记为违规：

"""
        for instance_id, result in cheating_instances.items():
            md_content += f"### {instance_id}\n\n"
            cheating_details = result.get('cheating_details', [])

            if cheating_details:
                md_content += f"**检测到 {len(cheating_details)} 项违规行为**：\n\n"

                # 按规则类型分组显示
                violations_by_category = {}
                for detail in cheating_details:
                    category = detail['rule_category']
                    if category not in violations_by_category:
                        violations_by_category[category] = []
                    violations_by_category[category].append(detail)

                for category, violations in violations_by_category.items():
                    category_names = {
                        'forbidden_system_access': '🚨 禁止系统访问',
                        'forbidden_commands': '🚫 禁止命令',
                        'forbidden_file_access': '📄 禁止文件访问',
                        'suspicious_patterns': '⚠️ 可疑行为模式'
                    }
                    md_content += f"#### {category_names.get(category, category)}\n\n"

                    for violation in violations:
                        md_content += f"- **第{violation['line_number']}行**: {violation['violation_type']}\n"
                        md_content += f"  - 匹配模式: `{violation['pattern']}`\n"
                        md_content += f"  - 检测内容: `{violation['matched_content'][:100]}{'...' if len(violation['matched_content']) > 100 else ''}`\n\n"

            md_content += "---\n\n"

    # 添加失败测试的详细信息（如果有的话）
    failed_instances = {k: v for k, v in instance_results.items()
                       if (v.get('failed_tests', 0) > 0 or v.get('error_tests', 0) > 0) and not v.get('is_cheating', False)}

    if failed_instances:
        md_content += """## 失败测试详情

"""
        for instance_id, result in failed_instances.items():
            md_content += f"### {instance_id}\n\n"

            # 从 repo_content 中获取失败的测试详情
            repo_content = result.get('repo_content')
            if repo_content:
                test_cases = repo_content.get('test_cases', {})
                failures = repo_content.get('failures', {})

                # 显示失败的测试，用代码块包围
                failed_tests = [test_name for test_name, status in test_cases.items() if status == 'failed']
                if failed_tests:
                    md_content += "**失败的测试**:\n```\n"
                    for test_name in failed_tests:
                        error_msg = failures.get(test_name, '无错误信息')
                        md_content += f"- {test_name}: {error_msg}\n"
                    md_content += "```\n\n"

    # 写入报告文件
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    if logger:
        logger.info(f"Report generated: {summary_md_path}")
        logger.info(f"Overall statistics: {resolved_count}/{total_instances} instances resolved ({resolution_rate:.1f}%)")
        if cheating_count > 0:
            logger.warning(f"Cheating detection: {cheating_count}/{total_instances} instances detected as cheating ({cheating_rate:.1f}%)")


def _build_test_command(base_cmd: str, test_path: str, timeout: int, output_file: str = None, json_report_file: str = None) -> str:
    """
    根据基础测试命令构建完整的测试命令

    Args:
        base_cmd: 基础测试命令，如 'pytest -vs' 或 'python tests/runtests.py --verbosity 2'
        test_path: 测试路径
        timeout: 超时时间（秒）
        output_file: 输出文件路径（可选）
        json_report_file: JSON报告文件路径（可选）

    Returns:
        str: 完整的测试命令
    """
    # 检查是否是 pytest 命令 或者 Django 命令
    is_pytest = 'pytest' in base_cmd
    is_django_runtests = 'runtests.py' in base_cmd

    # 构建命令 - 简单字符串拼接
    if is_pytest:
        # pytest 命令：添加必要参数
        cmd = f"{base_cmd} {test_path} --color=no"

        if '--timeout' not in base_cmd:
            cmd += f" --timeout={timeout}"

        if json_report_file:
            cmd += f" --json-report --json-report-file={json_report_file}"
        elif output_file:
            cmd += f" > {output_file} 2>&1"

    elif is_django_runtests:
        # Django runtests：保持原有命令
        cmd = base_cmd

        # 对于 Django，如果提供了测试路径，通常是测试模块名，直接添加
        if test_path:
            # FIXME: 这里应该是点分形式才对, 之后修
            cmd += f" {test_path}"

        # 添加输出重定向
        if output_file:
            cmd += f" > {output_file} 2>&1"

    else:
        raise ValueError(f"评测 pipeline 还未支持的测试命令: {base_cmd}")

    return cmd


def run_pytest_and_evaluate(runtime, instance, work_dir: str = 'workspace', test_time: int = 3600, logger=None, task_level=None, repo_name=None):
    """
    运行测试并生成评估结果的通用函数

    Args:
        runtime: OpenHands runtime 实例
        instance: 包含 instance_id、timeout 和 test_cmd 的实例对象
        work_dir: 工作目录名称，默认为 'workspace'
        test_time: 硬超时时间（秒），默认为 3600 秒
        logger: 日志记录器
        task_level: 任务级别
        repo_name: 仓库名称

    Returns:
        dict: 包含执行结果的字典，格式为：
        {
            'status': 'success' | 'failure',
            'error': str | None,
            'instance_id': str,
            'detail': str,
        }

    Notes:
        - 支持自定义测试命令通过 instance.test_cmd，默认为 'pytest -vs'
        - 对于 pytest 命令会自动添加 --timeout、--color=no 等参数
        - 对于 Django runtests.py 等命令会智能处理，不添加不支持的参数
    """

    detail = ''
    is_level_1 = (task_level == 1 or task_level == '1')

    # 获取测试命令，默认为 'pytest -vs'
    test_cmd_raw = getattr(instance, 'test_cmd', instance.get('test_cmd') if hasattr(instance, 'get') else None)

    # 处理各种无效值情况：None, NaN, 空字符串等
    if (test_cmd_raw is None or
        test_cmd_raw == '' or
        (isinstance(test_cmd_raw, float) and (str(test_cmd_raw).lower() == 'nan' or pd.isna(test_cmd_raw)))):
        test_cmd = 'pytest -vs'
    elif isinstance(test_cmd_raw, str):
        test_cmd = test_cmd_raw.strip() or 'pytest -vs'  # 处理只有空白字符的情况
    else:
        # 其他类型转换为字符串
        test_cmd = str(test_cmd_raw)

    logger.info(f"Using test command: {test_cmd}")

    # 1. 把测试目录挂载到 runtime 里
    sandbox_app_path = f'/{work_dir}'
    test_host_path = getattr(instance, 'test_path', instance.get('test_path') if hasattr(instance, 'get') else None)
    if test_host_path and os.path.exists(test_host_path):
        runtime.copy_to(test_host_path, sandbox_app_path, recursive=True)
        logger.info(f"Copied test from '{test_host_path}' to sandbox at '{sandbox_app_path}'")
    else:
        error_msg = f'Test path not provided or not found: {test_host_path}'
        logger.warning(error_msg)
        return {
            'status': 'failure',
            'error': error_msg,
            'instance_id': getattr(instance, 'instance_id', str(instance)),
            'detail': detail
        }

    # 2.1 读取 WORK_DIR/test/path2test.txt, 里面的 py 路径记作 test_path, 转为绝对路径
    sandbox_test_path = os.path.join(sandbox_app_path, 'test')
    path2test_file = os.path.join(sandbox_test_path, 'path2test.txt')

    # 2.2 检查 path2test.txt 是否存在
    action = CmdRunAction(command=f'test -f {path2test_file}')
    obs = runtime.run_action(action)
    if obs.exit_code != 0:
        error_msg = f"path2test.txt not found at {path2test_file}"
        logger.error(error_msg)
        return {
            'status': 'failure',
            'error': error_msg,
            'instance_id': getattr(instance, 'instance_id', str(instance)),
            'detail': detail
        }

    # 2.3 读取 path2test.txt 获取测试路径
    action = CmdRunAction(command=f'cat {path2test_file}')
    obs = runtime.run_action(action)
    if obs.exit_code != 0:
        error_msg = f"Failed to read path2test.txt: {obs.content}"
        logger.error(error_msg)
        return {
            'status': 'failure',
            'error': error_msg,
            'instance_id': getattr(instance, 'instance_id', str(instance)),
            'detail': detail
        }

    # 支持 path2test.txt 多行，每行一个测试路径
    test_path_lines = [line.strip() for line in obs.content.splitlines() if line.strip()]
    if not test_path_lines:
        error_msg = "path2test.txt is empty"
        logger.error(error_msg)
        return {
            'status': 'failure',
            'error': error_msg,
            'instance_id': getattr(instance, 'instance_id', str(instance)),
            'detail': detail
        }
    # test_path_list: 每个元素为一个测试路径
    test_path_list = test_path_lines

    # 对于 level 3, 需要先把 /testbed 下的东西清空, 然后把 sandbox_test_path/repo_name/* 里面的内容 mv 到 /testbed 下, 然后删除空文件夹 sandbox_test_path/repo_name
    if not is_level_1:  # level 3
        logger.info("[LV3] Starting level 3 specific processing")

        # 1. 清空 /testbed 下的所有内容
        clear_action = CmdRunAction(command='rm -rf /testbed/*')
        clear_obs = runtime.run_action(clear_action)
        if clear_obs.exit_code == 0:
            logger.info("[LV3] Successfully cleared /testbed directory")
        else:
            logger.warning(f"[LV3] Failed to clear /testbed directory: {clear_obs.content}")

        # 2. 将 sandbox_test_path/repo_name/* 的内容移动到 /testbed 下
        repo_source_path = os.path.join(sandbox_test_path, repo_name)
        move_action = CmdRunAction(command=f'mv {repo_source_path}/* /testbed/')
        move_obs = runtime.run_action(move_action)
        if move_obs.exit_code == 0:
            logger.info(f"[LV3] Successfully moved contents from {repo_source_path} to /testbed")
        else:
            logger.warning(f"[LV3] Failed to move contents from {repo_source_path} to /testbed: {move_obs.content}")

        # 4. 删除 sandbox_test_path/repo_name 文件夹
        rm_action = CmdRunAction(command=f'rm -rf {repo_source_path}')
        rm_obs = runtime.run_action(rm_action)
        if rm_obs.exit_code == 0:
            logger.info(f"[LV3] Successfully removed empty directory {repo_source_path}")
        else:
            logger.warning(f"[LV3] Failed to remove directory {repo_source_path}: {rm_obs.content}")

    # 对于 level 1, 需要对每个 test_path 进行绝对路径转换和移动
    if is_level_1:
        # level=1: test_path 形如 repo_name/../.., 需要替换为 testbed/...
        # 1. 对每个 test_path 进行路径转换和移动
        abs_test_path_list = []
        for test_path in test_path_list:
            test_name = os.path.basename(test_path) # 拿到测试文件自己的名字
            source_path = f'/{work_dir}/test/{test_name}'
            # 将 test_path 头部的 repo_name 替换为 testbed
            modified_test_path = test_path.replace(repo_name + '/', 'testbed/', 1)
            # target_test_path 就是 /{modified_test_path}
            target_test_path = f'/{modified_test_path}'
            action = CmdRunAction(command=f'mv {source_path} {target_test_path}')
            obs_mv = runtime.run_action(action)
            if obs_mv.exit_code != 0:
                logger.warning(f"Failed to move test file from {source_path} to {target_test_path}: {obs_mv.content}")
            abs_test_path_list.append(target_test_path)
        test_path_list = abs_test_path_list
    else:
        # level=3: 现在仓库内容已经移动到 /testbed 下
        # 需要把 test_path 转换为 /testbed 下的绝对路径
        abs_test_path_list = []
        for test_path in test_path_list:
            # 对于相对路径，需要从 test_path 中去掉 repo_name 前缀，然后拼接到 /testbed 下
            relative_path = test_path[len(repo_name) + 1:]
            # 拼接到 /testbed/ 下
            abs_test_path = os.path.join('/testbed', relative_path)
            abs_test_path_list.append(abs_test_path)
        test_path_list = abs_test_path_list

    logger.info(f"Test paths from path2test.txt: {test_path_list}")

    # 执行 /work_dir/test 下的 wrap_imports_with_try.py
    wrap_target_path = '/testbed'
    action = CmdRunAction(
        command=f'python {sandbox_test_path}/wrap_imports_with_try.py {wrap_target_path} -r --no-backup',
        blocking=True,  # 设置阻塞模式保证命令执行完成后再继续
    )
    action.set_hard_timeout(test_time)
    obs = runtime.run_action(action)
    if obs.exit_code != 0:
        detail += '  ' + obs.message
        print(obs.message)
    logger.info(f"Successfully ran wrap_imports_with_try.py on {wrap_target_path}")

    # 获取 timeout 值
    if hasattr(instance, 'timeout'):
        timeout = instance.timeout
    elif hasattr(instance, 'get') and 'timeout' in instance:
        timeout = instance['timeout']
    else:
        raise ValueError(f"Instance {getattr(instance, 'instance_id', str(instance))} does not have timeout attribute")

    # 2.4 针对每个测试文件分别执行 pytest，结果分别保存
    work_dir_path = '/testbed'

    repos_dir = os.path.join(sandbox_test_path, 'repos')
    raw_output_dir = os.path.join(sandbox_test_path, 'raw_output')
    # 创建结果目录
    runtime.run_action(CmdRunAction(command=f'mkdir -p {repos_dir}'))
    runtime.run_action(CmdRunAction(command=f'mkdir -p {raw_output_dir}'))

    first_repo_json_path = None
    for idx, test_path in enumerate(test_path_list):
        test_name = os.path.basename(test_path)
        raw_output_file = os.path.join(raw_output_dir, f'{test_name}.txt')
        repo_tmp = os.path.join(repos_dir, f'{test_name}.json')

        # 1. 执行测试并输出到文件
        test_command1 = _build_test_command(test_cmd, test_path, timeout, output_file=raw_output_file)
        action1 = CmdRunAction(
            command=f'cd {work_dir_path} && {test_command1}',
            blocking=True,
        )
        action1.set_hard_timeout(test_time)
        obs1 = runtime.run_action(action1)
        if obs1.exit_code == -1:
            detail += obs1.message
            print(obs1.message)
        logger.info(f"Test for {test_name} completed, output saved to {raw_output_file}")

        # 检查 raw_output_file 是否存在, 若不存在, continue
        check_action = CmdRunAction(command=f'test -f {raw_output_file}')
        check_obs = runtime.run_action(check_action)
        if check_obs.exit_code != 0:
            error_msg = f"Raw output file {raw_output_file} was not created"
            logger.error(error_msg)
            continue

        # 2. 执行测试生成 JSON 报告（仅对支持的命令），生成之前先进行一步静默删除 repo_tmp.json
        runtime.run_action(CmdRunAction(command=f'rm -f {os.path.join(sandbox_test_path, "repo_tmp.json")}'))
        repo_tmp_local = os.path.join(sandbox_test_path, 'repo_tmp.json')

        # 检查是否支持 JSON 报告（主要是 pytest）
        if 'pytest' in test_cmd:
            test_command2 = _build_test_command(test_cmd, test_path, timeout, json_report_file=repo_tmp_local)
            action2 = CmdRunAction(
                command=f'cd {work_dir_path} && {test_command2}',
                blocking=True,
            )
            action2.set_hard_timeout(test_time)
            obs2 = runtime.run_action(action2)
            if obs2.exit_code != 0:
                detail += f" {obs2.message}"
                logger.warning(f"Test json command failed for {test_name}: exit_code={obs2.exit_code}, content={obs2.content}")
                print(obs2.message)
            logger.info(f"Test json for {test_name} completed")
        # 运行到这里的就是 Django
        else:
            # 对于不支持 JSON 报告的命令，创建一个空的 JSON 文件或跳过
            logger.warning(f"Test command '{test_cmd}' does not support JSON reporting, skipping JSON report generation")
            # 创建一个最小的 JSON 报告结构，使用沙盒命令
            minimal_report_content = '''{
    "summary": {"total": 0, "passed": 0, "failed": 0, "error": 0},
    "test_cases": {},
    "failures": {},
    "note": "JSON reporting not supported for this test command"
}'''
            create_json_action = CmdRunAction(command=f'echo \'{minimal_report_content}\' > {repo_tmp_local}')
            runtime.run_action(create_json_action)

        # 检查 repo_tmp.json 文件是否存在, 若不存在, continue
        check_action2 = CmdRunAction(command=f'test -f {repo_tmp_local}')
        check_obs2 = runtime.run_action(check_action2)
        if check_obs2.exit_code != 0:
            error_msg = f"JSON report file {repo_tmp_local} was not created"
            logger.error(error_msg)
            continue

        # FIXME: 这里应该换为解析器到时候
        # 3. 执行 eval_code.py，生成 repo.json, 生成之前先进行一步静默删除 repo.json
        runtime.run_action(CmdRunAction(command=f'rm -f {os.path.join(sandbox_test_path, "repo.json")}'))
        action_eval = CmdRunAction(
            command=f'python {sandbox_test_path}/eval_code.py',
            blocking=True,
        )
        action_eval.set_hard_timeout(test_time)
        obs_eval = runtime.run_action(action_eval)
        if obs_eval.exit_code != 0:
            detail += '  ' + obs_eval.message
            print(obs_eval.message)
        logger.info(f"Successfully ran eval_code.py for {test_name}")

        # 检查 repo.json 是否生成
        repo_json_path = os.path.join(sandbox_test_path, 'repo.json')
        check_action3 = CmdRunAction(command=f'test -f {repo_json_path}')
        check_obs3 = runtime.run_action(check_action3)
        if check_obs3.exit_code != 0:
            error_msg = f"repo.json was not created for {test_name}"
            logger.error(error_msg)
            continue

        # 4. 移动 repo.json 到 repos/<test_name>.json
        move_action = CmdRunAction(command=f'mv {repo_json_path} {repo_tmp}')
        move_obs = runtime.run_action(move_action)
        if move_obs.exit_code != 0:
            logger.warning(f"Failed to move repo.json to {repo_tmp}: {move_obs.content}")

        # 记录第一个测试的 repo.json 路径
        if idx == 0:
            first_repo_json_path = repo_tmp

    # 兼容：将第一个测试的结果拷贝为 /work_dir/test/repo.json
    if first_repo_json_path:
        repo_json_link = os.path.join(sandbox_test_path, 'repo.json')
        runtime.run_action(CmdRunAction(command=f'cp {first_repo_json_path} {repo_json_link}'))

    return {
        'status': 'success',
        'error': None,
        'instance_id': getattr(instance, 'instance_id', str(instance)),
        'detail': detail,
    }


def extract_repo_json_and_cleanup(runtime, work_dir: str = 'workspace', logger=None):
    """
    提取 repo.json 文件内容并删除 test 目录

    Args:
        runtime: OpenHands runtime 实例
        work_dir: 工作目录名称，默认为 'workspace'
        logger: 日志记录器

    Returns:
        str: repo.json 文件的内容，如果出错则返回错误信息
    """
    repo_json_content = ""
    ok = 0
    try:
        # 1. 读 repo.json 文件
        repo_json_path = f"/{work_dir}/test/repo.json"
        read_action = FileReadAction(path=repo_json_path)
        obs = runtime.run_action(read_action)

        if hasattr(obs, 'content') and obs.content:
            repo_json_content = obs.content
            ok += 1
            logger.info(f"成功读取 repo.json 文件，内容长度: {len(repo_json_content)}")
        else:
            logger.warning(f"无法读取 repo.json 文件: {obs}")
            repo_json_content = "Unable to read repo.json file content"

        # 2. 删除 /work_dir/test 目录
        delete_action = CmdRunAction(command=f"rm -rf /{work_dir}/test")
        delete_obs = runtime.run_action(delete_action)
        if delete_obs.exit_code == 0:
            ok += 1
            logger.info(f"成功删除 /{work_dir}/test 目录")
        else:
            logger.warning(f"删除 /{work_dir}/test 目录失败: {delete_obs.content}")

    except Exception as e:
        logger.warning(f"处理 repo.json 或删除目录时出错: {e}")
        repo_json_content = f"Error during processing: {str(e)}"

    return repo_json_content, (ok == 2)


def cleanup_test_directory(runtime, work_dir: str = 'workspace'):
    """
    静默删除 test 目录进行清理

    Args:
        runtime: OpenHands runtime 实例
        work_dir: 工作目录名称，默认为 'workspace'

    Raises:
        Exception: 当目录存在但删除失败时
    """
    # 先检查目录是否存在
    check_action = CmdRunAction(command=f"test -d /{work_dir}/test")
    check_obs = runtime.run_action(check_action)

    # 如果目录不存在，直接返回
    if check_obs.exit_code != 0:
        return

    # 目录存在，尝试删除
    delete_action = CmdRunAction(command=f"rm -rf /{work_dir}/test")
    delete_obs = runtime.run_action(delete_action)

    # 如果删除失败，抛出异常
    if delete_obs.exit_code != 0:
        raise RuntimeError(f"Failed to delete /{work_dir}/test directory: {delete_obs.content}")
