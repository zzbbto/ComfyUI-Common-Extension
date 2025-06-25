import importlib
import sys
import os
import traceback
from pathlib import Path

# 获取当前脚本所在目录
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_directory)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 自动发现scripts目录及其所有子目录下的节点模块
scripts_dir = os.path.join(script_directory, "scripts")

if os.path.exists(scripts_dir):
    # 使用pathlib递归查找所有Python文件
    py_files = Path(scripts_dir).rglob("*.py")
    
    for py_file in py_files:
        # 跳过特殊文件和__pycache__目录
        if py_file.name.startswith("__") or "__pycache__" in py_file.parts:
            continue
            
        # 转换为模块路径 (scripts/subdir/module => scripts.subdir.module)
        rel_path = py_file.relative_to(script_directory)
        module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
        
        try:
            # 动态导入模块
            imported_module = importlib.import_module(module_path)
            
            # 合并节点映射
            if hasattr(imported_module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
                
            if hasattr(imported_module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
                
        except Exception as e:
            print(f"🚨 Failed to import module: {module_path}")
            traceback.print_exc()
else:
    print(f"⚠️ Scripts directory not found: {scripts_dir}")

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]