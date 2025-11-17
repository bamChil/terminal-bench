import os
import shutil
from pathlib import Path
from typing import Optional


def replace_with_masked_code(target_dir: str, script_dir: Optional[str] = None):
    """
    读取script_dir下的notes和origin文件夹，并依据case_converter.py的相同逻辑
    替换target_dir下的某些文件。
    
    Args:
        target_dir: 目标目录路径，需要被替换文件的目录
        script_dir: 脚本所在目录，包含notes和origin文件夹，默认为当前脚本所在目录
    """
    if script_dir is None:
        script_dir = Path(__file__).parent
    else:
        script_dir = Path(script_dir)
    
    target_dir = Path(target_dir)
    notes_dir = script_dir / "notes"
    origin_dir = script_dir / "origin"
    
    # 检查必要的目录是否存在
    if not target_dir.exists():
        raise ValueError(f"目标目录不存在: {target_dir}")
    
    if not notes_dir.exists() or not notes_dir.is_dir():
        raise ValueError(f"notes目录不存在: {notes_dir}")
    

    origin_only_files = get_origin_only_files(notes_dir, origin_dir)

    # 遍历notes目录下的所有.py文件
    replaced_count = 0
    for note_file in notes_dir.iterdir():
        if note_file.is_file() and note_file.name.endswith(".py"):
            # 将文件名中的-替换成/，构建相对路径
            # 例如: src-liger-cross.py -> src/liger/cross.py
            rel_path = Path(str(note_file.name).replace("-", "/"))
            
            # 目标文件路径
            target_file_path = target_dir / rel_path
            
            # 复制文件
            try:
                shutil.copy2(note_file, target_file_path)
                print(f"已替换: {rel_path}")
                replaced_count += 1
            except Exception as e:
                print(f"替换文件失败 {note_file.name} -> {rel_path}: {e}")
    
    # 删除notes中有但origin中没有的文件对应的target_dir中的文件
    deleted_count = 0
    for origin_only_file in origin_only_files:
        # 将文件名中的-替换成/，构建相对路径
        rel_path = Path(str(origin_only_file).replace("-", "/"))
        
        # 目标文件路径
        target_file_path = target_dir / rel_path
        
        # 删除文件
        if target_file_path.exists():
            try:
                target_file_path.unlink()
                print(f"已删除: {rel_path}")
                deleted_count += 1
            except Exception as e:
                print(f"删除文件失败 {rel_path}: {e}")
        else:
            print(f"文件不存在，跳过删除: {rel_path}")
    
    print(f"完成替换，共处理了 {replaced_count} 个文件")
    print(f"完成删除，共删除了 {deleted_count} 个文件")


def get_origin_only_files(notes_dir: Path, origin_dir: Path):
    """
    验证notes和origin目录中文件的对应关系
    返回匹配的文件和notes独有的文件
    """
    notes_files = {f.name for f in notes_dir.iterdir() if f.is_file() and f.name.endswith(".py")}
    origin_files = {f.name for f in origin_dir.iterdir() if f.is_file() and f.name.endswith(".py")}
    
    # origin中有但notes中没有的文件
    origin_only_files = origin_files - notes_files
    if origin_only_files:
        print(f"origin中独有的文件（将删除对应的target文件）: {sorted(origin_only_files)}")
    
    return origin_only_files


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python replace_with_masked_code.py <target_directory>")
        print("示例: python replace_with_masked_code.py /testbed/")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    
    try:
        replace_with_masked_code(target_directory)
        print("替换完成!")
    except Exception as e:
        print(f"执行失败: {e}")
        sys.exit(1)
