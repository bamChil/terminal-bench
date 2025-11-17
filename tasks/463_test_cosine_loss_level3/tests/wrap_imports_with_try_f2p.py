#!/usr/bin/env python3
"""
专门用于 f2p 后验测试的 import 包裹脚本
根据 AST 解析结果，对匹配的 import 语句进行 try/except 包裹（支持所有缩进级别）
"""
import sys
import os
import ast


def main():
    if len(sys.argv) < 3:
        print("Usage: wrap_imports_with_try_f2p.py <test_file> <module1> [<module2> ...]", file=sys.stderr)
        sys.exit(1)

    test_file = sys.argv[1]
    target_modules = sys.argv[2:]  # 点分形式的模块名列表，如 ['transformers.models.bert', 'torch.nn']

    if not os.path.exists(test_file):
        print(f"[wrap_imports_f2p] test file not found: {test_file}", file=sys.stderr)
        return 0

    # 读取原始内容
    with open(test_file, 'r', encoding='utf-8') as f:
        src_content = f.read()

    try:
        # 使用 AST 解析源代码
        tree = ast.parse(src_content, filename=test_file)
    except SyntaxError as e:
        print(f"[wrap_imports_f2p] syntax error in {test_file}: {e}", file=sys.stderr)
        return 1

    # 找到所有 import 语句并提取其对应的点分形式（不限制缩进级别）
    import_nodes_to_wrap = []
    
    for node in ast.walk(tree):
        # 处理所有的 Import 和 ImportFrom 节点，不论缩进级别
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            dotted_names = extract_dotted_names_from_import(node)
            
            # 检查是否有任何一个点分名称在目标列表中
            for dotted_name in dotted_names:
                if should_wrap_import(dotted_name, target_modules):
                    import_nodes_to_wrap.append(node)
                    break  # 找到一个匹配即可，不用继续检查这个 import 的其他名称

    if not import_nodes_to_wrap:
        print(f"[wrap_imports_f2p] no matching imports found in {test_file}", file=sys.stderr)
        return 0

    # 根据行号信息将匹配的 import 语句包裹在 try/except 中
    src_lines = src_content.splitlines(keepends=True)  # 保持换行符
    
    # 收集需要包裹的import信息，并过滤已包装和特殊import
    import_infos_to_wrap = []
    for node in import_nodes_to_wrap:
        start_line = node.lineno - 1  # AST 行号从 1 开始，转换为 0 开始的索引
        
        # 检查是否已经被包装
        if is_already_wrapped(src_lines, start_line):
            continue
            
        # 获取完整的import行（包括多行import）
        import_lines = get_import_block_lines(src_lines, start_line, node)
        
        # 检查是否应该跳过这个import（如 __future__ import）
        if should_skip_import(import_lines):
            continue
            
        import_infos_to_wrap.append((start_line, import_lines))
    
    if not import_infos_to_wrap:
        print(f"[wrap_imports_f2p] no valid imports to wrap in {test_file}", file=sys.stderr)
        return 0
    
    # 从后往前处理，避免行号偏移问题
    import_infos_to_wrap.sort(key=lambda x: x[0], reverse=True)
    
    new_lines = src_lines.copy()
    wrapped_count = 0
    
    for start_line, import_lines in import_infos_to_wrap:
        # 获取原始缩进
        original_indent = get_indentation(import_lines[0])
        
        # 包装import语句
        wrapped_lines = wrap_import_with_try(import_lines, original_indent)
        
        # 删除原来的import行
        for _ in range(len(import_lines)):
            if start_line < len(new_lines):
                new_lines.pop(start_line)
        
        # 插入包装后的行
        for i, wrapped_line in enumerate(wrapped_lines):
            new_lines.insert(start_line + i, wrapped_line)
        
        wrapped_count += 1

    # 写回文件
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"[wrap_imports_f2p] wrapped {wrapped_count} import statements in {test_file}")
    return 0


def is_already_wrapped(src_lines, line_idx):
    """检查导入语句是否已经被 try-except 包装"""
    # 向前查找几行，看是否有 try:
    for i in range(max(0, line_idx - 3), line_idx):
        if i < len(src_lines):
            line = src_lines[i].strip()
            if line == 'try:' or line.startswith('try:'):
                return True
    return False


def get_import_block_lines(src_lines, start_line, node):
    """获取导入语句的所有行（包括续行）"""
    end_line = getattr(node, 'end_lineno', node.lineno) - 1
    
    # 基本的行范围
    import_lines = []
    for i in range(start_line, min(end_line + 1, len(src_lines))):
        import_lines.append(src_lines[i])
    
    # 检查是否有续行（以 \ 结尾或在括号内）
    if import_lines:
        last_line = import_lines[-1]
        stripped = last_line.strip()
        
        # 如果最后一行以 \ 结尾或者括号不匹配，继续查找
        if (stripped.endswith('\\') or 
            stripped.count('(') > stripped.count(')') or
            stripped.count('[') > stripped.count(']') or
            stripped.count('{') > stripped.count('}')):
            
            # 继续查找直到导入语句完整
            j = end_line + 1
            while j < len(src_lines):
                next_line = src_lines[j]
                import_lines.append(next_line)
                next_stripped = next_line.strip()
                
                # 检查是否导入语句结束
                if (not next_stripped.endswith('\\') and
                    next_stripped.count('(') <= next_stripped.count(')') and
                    next_stripped.count('[') <= next_stripped.count(']') and
                    next_stripped.count('{') <= next_stripped.count('}')):
                    break
                j += 1
    
    return import_lines


def should_skip_import(import_lines):
    """检查是否应该跳过这个导入语句"""
    # 检查是否是 __future__ 导入
    for line in import_lines:
        stripped_line = line.strip()
        if stripped_line.startswith('from __future__'):
            return True
    return False


def get_indentation(line):
    """获取行的缩进"""
    if isinstance(line, str):
        return line[:len(line) - len(line.lstrip())]
    return ""


def wrap_import_with_try(import_lines, original_indent):
    """将导入语句包装在 try-except 中"""
    wrapped_lines = []
    
    # try 行
    wrapped_lines.append(f"{original_indent}try:\n")
    
    # 导入语句行（增加缩进）
    for line in import_lines:
        if line.strip():  # 跳过空行
            # 为导入语句添加额外的缩进
            new_line = f"{original_indent}    {line.lstrip()}"
            # 如果原行末尾没有换行符，添加一个
            if not new_line.endswith('\n'):
                new_line += '\n'
            wrapped_lines.append(new_line)
        else:
            wrapped_lines.append(line)
    
    # except 行
    wrapped_lines.append(f"{original_indent}except Exception as _e:\n")
    wrapped_lines.append(f"{original_indent}    # Defer import errors to test runtime to not break discovery\n")
    wrapped_lines.append(f"{original_indent}    _e = _e  # no-op to silence linters\n")
    
    return wrapped_lines


def extract_dotted_names_from_import(node):
    """从 import 节点中提取所有点分名称（与 _get_imports 逻辑保持一致）"""
    dotted_names = []
    
    if isinstance(node, ast.Import):
        # import torch.nn → 提取 torch.nn
        for alias in node.names:
            dotted_names.append(alias.name)
    
    elif isinstance(node, ast.ImportFrom):
        # from transformers.models import bert → 提取 transformers.models.bert
        if node.module:
            base_module = node.module
            for alias in node.names:
                if alias.name == '*':
                    # from module import * → 提取 module
                    dotted_names.append(base_module)
                else:
                    # from a.b import c → 提取 a.b.c
                    full_name = f"{base_module}.{alias.name}"
                    dotted_names.append(full_name)
    
    return dotted_names


def should_wrap_import(dotted_name, target_modules):
    """检查点分名称是否在目标列表中（精确匹配或前缀匹配）"""
    # 精确匹配
    if dotted_name in target_modules:
        return True
    
    # 前缀匹配：检查 dotted_name 是否以任何目标模块开头
    # 例如：transformers.AutoTokenizer 匹配 transformers
    for target_module in target_modules:
        if dotted_name.startswith(target_module + '.'):
            return True
    
    return False


if __name__ == '__main__':
    sys.exit(main())