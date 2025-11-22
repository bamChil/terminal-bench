'''
总结run_id下的results.json文件，
主要关注了每个任务的通过率
用法:

```bash
python scripts_python/sum_results.py runs/2025-11-17__18-11-41/results.json
```

'''

import json
import sys
from pathlib import Path

def calculate_stats(parser_results):
    """计算passed和failed数量"""
    if parser_results is None or not isinstance(parser_results, dict):
        return None, None, None
    
    passed = 0
    failed = 0
    
    for test_name, status in parser_results.items():
        if status == "passed":
            passed += 1
        elif status == "failed":
            failed += 1
    
    total = passed + failed
    passed_rate = passed / total if total > 0 else 0
    return passed_rate, passed, failed

def process_results(input_file):
    """处理results.json文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_data = {
        "results": []
    }
    
    # 获取results数组
    results_array = data.get("results", [])
    
    # 遍历原始数据中的每个任务
    for task_data in results_array:
        if not isinstance(task_data, dict):
            continue
            
        parser_results = task_data.get("parser_results")
        passed_rate, passed, failed = calculate_stats(parser_results)
        
        result_item = {
            "id": task_data.get("task_id"),
            "trial_name": task_data.get("trial_name"),
            "failure_mode": task_data.get("failure_mode"),
            "passed_rate": passed_rate,
            "passed": passed,
            "failed": failed
        }
        output_data["results"].append(result_item)
    
    return output_data

def main():
    if len(sys.argv) != 2:
        print("使用方法: python sum_results.py <输入文件路径>")
        print("示例: python sum_results.py runs/2025-11-17__18-11-41/results.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not Path(input_file).exists():
        print(f"错误: 文件 '{input_file}' 不存在")
        sys.exit(1)
    
    print(f"正在处理文件: {input_file}")
    
    # 处理数据
    output_data = process_results(input_file)
    
    # 生成输出文件名
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}stats.json"
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"统计完成! 结果已保存到: {output_file}")
    print(f"总共处理了 {len(output_data['results'])} 个任务")
    
    # 打印一些统计信息
    total_tasks = len(output_data['results'])
    tasks_with_results = sum(1 for r in output_data['results'] if r['passed'] is not None)
    tasks_without_results = total_tasks - tasks_with_results
    
    print(f"- 有测试结果的任务: {tasks_with_results}")
    print(f"- 无测试结果的任务: {tasks_without_results}")

if __name__ == "__main__":
    main()