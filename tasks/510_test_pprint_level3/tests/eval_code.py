import json
import re
from pathlib import Path
import argparse

def transform_pytest_report(input_data):
    # 处理 summary
    passed = input_data.get("summary", {}).get("passed", 0)
    failed = input_data.get("summary", {}).get("failed", 0)
    skipped = input_data.get("summary", {}).get("skipped", 0)
    deselected = input_data.get("summary", {}).get("deselected", 0)
    xfailed = input_data.get("summary", {}).get("xfailed", 0)
    xpassed = input_data.get("summary", {}).get("xpassed", 0)
    exitcode = input_data.get("exitcode", 0)
    collected = input_data.get("summary", {}).get("collected", 0)

    # 计算实际的错误数量，排除已知的状态
    known_states = passed + failed + skipped + deselected + xfailed + xpassed
    error = max(0, collected - known_states)

    summary = {
        "total": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "error": error,
        "return_code": exitcode,
        "success": exitcode == 0
    }

    # 处理 test cases 和 failures
    test_cases = {}
    failures = {}

    # 处理正常的测试用例
    for test in input_data.get("tests", []):
        nodeid = test.get("nodeid")
        outcome = test.get("outcome")

        if nodeid and outcome:
            test_cases[nodeid] = outcome

            # 只有真正失败的测试才记录到 failures 中
            if outcome == "failed":
                # 提取失败信息
                failure_msg = "Unknown error"
                call_data = test.get("call", {})

                if "crash" in call_data:
                    failure_msg = call_data["crash"].get("message", failure_msg)
                elif "longrepr" in call_data:
                    # 提取最后一行以 'E ' 开头的内容
                    lines = call_data["longrepr"].split('\n')
                    for line in reversed(lines):
                        if line.startswith('E '):
                            failure_msg = line[2:].strip()
                            break

                failures[nodeid] = failure_msg

    # 处理 collectors 中的导入错误（当 tests 为空但有收集器失败时）
    if not input_data.get("tests") and input_data.get("collectors"):
        for collector in input_data.get("collectors", []):
            if collector.get("outcome") == "failed":
                nodeid = collector.get("nodeid", "")
                longrepr = collector.get("longrepr", "")
                
                # 如果没有具体的文件路径作为 nodeid，尝试从 longrepr 中提取
                if not nodeid and longrepr:
                    # 通用的文件路径提取，寻找单引号包围的路径
                    match = re.search(r"'([^']*\.py)'", longrepr)
                    if match:
                        file_path = match.group(1)
                        # 移除常见的工作目录前缀，使路径更简洁
                        for prefix in ['/workspace/', '/tmp/', '/home/', '/usr/', '/opt/']:
                            if file_path.startswith(prefix):
                                # 找到第一个看起来像项目根目录的部分
                                parts = file_path[len(prefix):].split('/')
                                if len(parts) > 1:
                                    # 保留从第一个包含"test"的目录开始的路径，或者从第二个目录开始
                                    for i, part in enumerate(parts):
                                        if 'test' in part.lower() or i == 1:
                                            nodeid = '/'.join(parts[i:])
                                            break
                                    else:
                                        nodeid = '/'.join(parts[1:]) if len(parts) > 1 else parts[0]
                                break
                        else:
                            # 如果没有匹配的前缀，直接使用相对路径
                            nodeid = file_path
                
                # 如果还是没有nodeid，使用默认值
                if not nodeid:
                    nodeid = "collection_error"
                
                # 提取错误信息，优先提取以 'E ' 开头的最后一行
                failure_msg = "Collection failed"
                if longrepr:
                    lines = longrepr.split('\n')
                    for line in reversed(lines):
                        if line.startswith('E '):
                            failure_msg = line[2:].strip()
                            break
                    
                    # 如果没有找到 'E ' 开头的行，使用第一行非空行作为错误信息
                    if failure_msg == "Collection failed":
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('=') and not line.startswith('-'):
                                failure_msg = line
                                break
                
                test_cases[nodeid] = "failed"
                failures[nodeid] = failure_msg

    # 如果完全没有测试用例但有错误退出码
    if not test_cases and exitcode != 0:
        test_cases["pytest_execution_error"] = "failed"
        failures["pytest_execution_error"] = f"Pytest execution failed with exit code {exitcode}"
        
        # 更新 summary 以反映实际情况
        summary["total"] = 0
        summary["failed"] = 0
        summary["skipped"] = 0
        summary["error"] = 0

    # 返回转换后的报告
    return {
        "summary": summary,
        "test_cases": test_cases,
        "failures": failures
    }

def main():
    parser = argparse.ArgumentParser(description="转换 pytest 的 json 报告为标准格式")
    parser.add_argument('--input', type=str, default="repo_tmp.json", help="输入文件名，默认为 repo_tmp.json")
    parser.add_argument('--output', type=str, default="repo.json", help="输出文件名，默认为 repo.json")
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    input_file = current_dir / args.input
    output_file = current_dir / args.output

    # 检查输入文件是否存在
    if not input_file.exists():
        print("错误：未找到输入文件！")
        return

    try:
        # 读取输入 JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # 转换数据
        transformed_data = transform_pytest_report(original_data)

        # 写入输出 JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)

        print(f"成功生成 {args.output}")

    except json.JSONDecodeError:
        print("错误：输入文件不是有效的 JSON！")
    except Exception as e:
        print(f"发生异常: {str(e)}")

if __name__ == "__main__":
    main()
