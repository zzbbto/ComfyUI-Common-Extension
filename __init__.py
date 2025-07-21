import importlib
from pathlib import Path
import sys
import os

# 获取插件根目录
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_directory)

# 导入核心逻辑
try:
    from .lora_preview import setup_plugin as lora_setup_plugin
except ImportError as e:
    lora_setup_plugin = None

# 节点注册
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

# 注册路由
def register_routes():
    try:
        from server import PromptServer
        if hasattr(PromptServer, "instance") and PromptServer.instance is not None:
            server_instance = PromptServer.instance
            if hasattr(server_instance, "app"):
                # 检查路由表是否冻结（首次启动为 False）
                if not server_instance.app.router.frozen:
                    if lora_setup_plugin:
                        lora_setup_plugin(server_instance)
                        print("[CommonExtension] 首次启动：路由注册成功")
                else:
                    print("[CommonExtension] 路由表已冻结，跳过注册")
    except Exception as e:
        print(f"[CommonExtension] 注册失败：{e}")

# 首次启动时执行
if __name__ != "__main__":
    register_routes()

# 导出变量
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]