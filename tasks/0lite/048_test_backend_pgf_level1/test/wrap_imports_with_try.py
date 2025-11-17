#!/usr/bin/env python3
import ast
import os
import sys
import argparse
from typing import List, Tuple
import re

class ImportWrapper:
    def __init__(self, file_path: str, backup: bool = True):
        self.file_path = file_path
        self.backup = backup
        self.lines = []
        
    def read_file(self) -> List[str]:
        """读取文件内容"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
            return self.lines
        except Exception as e:
            raise RuntimeError(f"无法读取文件 {self.file_path}: {e}")
    
    def backup_file(self):
        """备份原文件"""
        if self.backup:
            backup_path = self.file_path + '.backup'
            try:
                with open(self.file_path, 'r', encoding='utf-8') as src:
                    with open(backup_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"⚠️  备份失败: {e}")
    
    def find_import_statements(self) -> List[Tuple[int, int, str]]:
        """
        使用 AST 找到所有导入语句
        返回: [(start_line, end_line, import_type), ...]
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f"无法解析 {self.file_path}: {e}")
        
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                
                # 处理多行导入的情况
                if end_line is None:
                    end_line = start_line
                
                import_type = 'import' if isinstance(node, ast.Import) else 'from_import'
                imports.append((start_line, end_line, import_type))
        
        # 按行号排序
        imports.sort(key=lambda x: x[0])
        return imports
    
    def get_import_block_lines(self, start_line: int, end_line: int) -> List[str]:
        """获取导入语句的所有行（包括续行）"""
        # 转换为 0 索引
        start_idx = start_line - 1
        end_idx = end_line - 1
        
        # 检查是否有续行（以 \ 或在括号内）
        lines = []
        for i in range(start_idx, min(end_idx + 1, len(self.lines))):
            line = self.lines[i]
            lines.append(line)
            
            # 如果行以 \ 结尾或者在括号/方括号/大括号内，继续查找
            stripped = line.strip()
            if (stripped.endswith('\\') or 
                stripped.count('(') > stripped.count(')') or
                stripped.count('[') > stripped.count(']') or
                stripped.count('{') > stripped.count('}')):
                
                # 继续查找直到导入语句完整
                j = i + 1
                while j < len(self.lines):
                    next_line = self.lines[j]
                    lines.append(next_line)
                    next_stripped = next_line.strip()
                    
                    # 检查是否导入语句结束
                    if (not next_stripped.endswith('\\') and
                        next_stripped.count('(') <= next_stripped.count(')') and
                        next_stripped.count('[') <= next_stripped.count(']') and
                        next_stripped.count('{') <= next_stripped.count('}')):
                        break
                    j += 1
                break
        
        return lines
    
    def is_already_wrapped(self, line_idx: int) -> bool:
        """检查导入语句是否已经被 try-except 包装"""
        # 向前查找几行，看是否有 try:
        for i in range(max(0, line_idx - 3), line_idx):
            if i < len(self.lines):
                line = self.lines[i].strip()
                if line == 'try:' or line.startswith('try:'):
                    return True
        return False
    
    def should_skip_import(self, import_lines: List[str]) -> bool:
        """检查是否应该跳过这个导入语句"""
        # 检查是否是 __future__ 导入
        for line in import_lines:
            stripped_line = line.strip()
            if stripped_line.startswith('from __future__'):
                return True
        return False
    
    def get_indentation(self, line: str) -> str:
        """获取行的缩进"""
        return line[:len(line) - len(line.lstrip())]
    
    def wrap_import_with_try(self, import_lines: List[str], original_indent: str) -> List[str]:
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
        wrapped_lines.append(f"{original_indent}except Exception:\n")
        wrapped_lines.append(f"{original_indent}    pass  # Import failed, continuing without this module\n")
        
        return wrapped_lines
    
    def process_file(self):
        """处理文件，添加 try-except 包装"""
        print(f"🔄 正在处理文件: {self.file_path}")
        
        # 读取文件
        lines = self.read_file()
        
        # 备份文件
        self.backup_file()
        
        # 找到所有导入语句
        imports = self.find_import_statements()
        
        if not imports:
            return
        
        # 从后往前处理，这样不会影响行号
        processed_count = 0
        new_lines = lines.copy()
        
        for start_line, end_line, import_type in reversed(imports):
            start_idx = start_line - 1
            
            # 获取导入语句的所有行
            import_lines = []
            for i in range(start_line - 1, min(end_line, len(new_lines))):
                if i < len(new_lines):
                    import_lines.append(new_lines[i])
            
            if not import_lines:
                continue
            
            # 检查是否应该跳过这个导入语句（如 __future__ 导入）
            if self.should_skip_import(import_lines):
                continue
            
            # 检查是否已经被包装
            if self.is_already_wrapped(start_idx):
                continue
            
            # 获取原始缩进
            original_indent = self.get_indentation(import_lines[0])
            
            # 包装导入语句
            wrapped_lines = self.wrap_import_with_try(import_lines, original_indent)
            
            # 替换原来的行
            # 删除原来的导入行
            for _ in range(len(import_lines)):
                if start_idx < len(new_lines):
                    new_lines.pop(start_idx)
            
            # 插入包装后的行
            for i, wrapped_line in enumerate(wrapped_lines):
                new_lines.insert(start_idx + i, wrapped_line)
            
            processed_count += 1
        
        # 写回文件
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"❌ 写文件失败: {e}")
            # 尝试恢复备份
            if self.backup:
                try:
                    backup_path = self.file_path + '.backup'
                    with open(backup_path, 'r', encoding='utf-8') as backup:
                        with open(self.file_path, 'w', encoding='utf-8') as original:
                            original.write(backup.read())
                    print("🔄 已从备份恢复原文件")
                except:
                    pass
            raise


def process_single_file(file_path: str, backup: bool = True):
    """处理单个文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    if not file_path.endswith('.py'):
        print(f"❌ 只能处理 Python 文件，跳过: {file_path}")
        return False
    
    try:
        wrapper = ImportWrapper(file_path, backup)
        wrapper.process_file()
        return True
    except Exception as e:
        print(f"❌ 处理文件失败 {file_path}: {e}")
        return False


def process_directory(dir_path: str, recursive: bool = False, backup: bool = True):
    """处理目录中的所有 Python 文件"""
    if not os.path.isdir(dir_path):
        print(f"❌ 目录不存在: {dir_path}")
        return
    
    print(f"📁 处理目录: {dir_path}")
    
    success_count = 0
    total_count = 0
    
    if recursive:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    total_count += 1
                    if process_single_file(file_path, backup):
                        success_count += 1
                    print()  # 空行分隔
    else:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) and file.endswith('.py'):
                total_count += 1
                if process_single_file(file_path, backup):
                    success_count += 1
                print()  # 空行分隔
    
    print(f"📊 目录处理完成: {success_count}/{total_count} 个文件处理成功")


def main():
    parser = argparse.ArgumentParser(
        description="自动为 Python 文件中的 import 语句添加 try-except 包装",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python wrap_imports_with_try.py file.py                    # 处理单个文件
  python wrap_imports_with_try.py /path/to/directory          # 处理目录中所有 .py 文件
  python wrap_imports_with_try.py /path/to/directory -r       # 递归处理目录
  python wrap_imports_with_try.py file.py --no-backup        # 不创建备份文件
        """)
    
    parser.add_argument('path', help='要处理的文件或目录路径')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='递归处理子目录中的文件')
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份文件')
    
    args = parser.parse_args()
    
    backup = not args.no_backup
    
    if os.path.isfile(args.path):
        print("🚀 开始处理单个文件...")
        process_single_file(args.path, backup)
    elif os.path.isdir(args.path):
        print("🚀 开始处理目录...")
        process_directory(args.path, args.recursive, backup)
    else:
        print(f"❌ 路径不存在: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
