import importlib
import traceback
from pathlib import Path
import sys
import os
import asyncio

# 获取插件根目录
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_directory)

# 导入 lora_preview 核心逻辑
try:
    from .lora_preview import setup_plugin as lora_setup_plugin
except ImportError as e:
    lora_setup_plugin = None

# 节点注册逻辑（保留原有节点加载）
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
scripts_dir = os.path.join(script_directory, "scripts_common")
if os.path.exists(scripts_dir):
    for py_file in Path(scripts_dir).rglob("*.py"):
        if py_file.name.startswith("__") or "__pycache__" in py_file.parts:
            continue
        rel_path = py_file.relative_to(script_directory)
        module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
        try:
            mod = importlib.import_module(module_path)
            NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
            NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        except Exception as e:
            print(f"[CommonExtension] 导入模块 {module_path} 失败：{e}")
else:
    print(f"[CommonExtension] 未找到 scripts_common 目录：{scripts_dir}")

# 注册路由（核心入口）
def delayed_register():
    try:
        from server import PromptServer
        if hasattr(PromptServer, "instance") and PromptServer.instance is not None:
            server_instance = PromptServer.instance
            if hasattr(server_instance, "app"):
                if lora_setup_plugin:
                    lora_setup_plugin(server_instance)  # 调用 lora_preview 的注册逻辑
                return
    except Exception as e:
        print(f"[CommonExtension] 路由注册失败 {e}")


# 启动延迟任务
if __name__ != "__main__":
    try:
        delayed_register()
    except Exception as e:
        print(f"[CommonExtension] {e}")

# 导出必要变量
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]