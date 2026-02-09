import importlib
import logging
import os
import sys
from pathlib import Path

# ===================== 基础配置 =====================
# 统一日志（对齐 workflow-to-api-converter 风格）
logger = logging.getLogger(__name__)
__version__ = "1.0.0"

# 获取插件根目录
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIRECTORY)

# ===================== 节点注册 =====================
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 导入 scripts_common 下的节点
scripts_dir = os.path.join(SCRIPT_DIRECTORY, "scripts_common")
if os.path.exists(scripts_dir):
    for py_file in Path(scripts_dir).rglob("*.py"):
        # 跳过特殊文件/目录
        if py_file.name.startswith("__") or "__pycache__" in py_file.parts:
            continue
        # 构建模块路径
        rel_path = py_file.relative_to(SCRIPT_DIRECTORY)
        module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
        try:
            mod = importlib.import_module(module_path)
            # 合并节点映射（容错处理）
            NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
            NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
            logger.info(f"[CommonExtension v{__version__}] 成功导入节点模块：{module_path}")
        except Exception as e:
            logger.error(f"[CommonExtension v{__version__}] 导入模块 {module_path} 失败：{str(e)}", exc_info=True)
else:
    logger.warning(f"[CommonExtension v{__version__}] 未找到 scripts_common 目录：{scripts_dir}")

# ===================== 路由注册总入口 =====================
def register_routes():
    """ComfyUI 启动时注册所有插件路由"""
    try:
        from server import PromptServer
        # 检查 PromptServer 实例
        if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
            logger.warning(f"[CommonExtension v{__version__}] PromptServer 实例未初始化，跳过路由注册")
            return
        
        server_instance = PromptServer.instance
        if not hasattr(server_instance, "app"):
            logger.warning(f"[CommonExtension v{__version__}] PromptServer 无 app 实例，跳过路由注册")
            return

        # 导入并初始化插件
        try:
            from .plugins.lora_preview import LoraPreviewPlugin
            from .plugins.sampler_scheduler import SamplerSchedulerPlugin
            from .plugins.remote_workflow import RemoteWorkflowPlugin  # 新增：导入远程工作流插件
            
            # 注册各插件路由
            LoraPreviewPlugin.register_routes(server_instance)
            SamplerSchedulerPlugin.register_routes(server_instance)
            RemoteWorkflowPlugin.register_routes(server_instance)
            
            logger.info(f"[CommonExtension v{__version__}] 所有插件路由注册完成")
        except ImportError as e:
            logger.error(f"[CommonExtension v{__version__}] 导入插件模块失败：{str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"[CommonExtension v{__version__}] 路由注册失败：{str(e)}", exc_info=True)

    except ImportError as e:
        logger.error(f"[CommonExtension v{__version__}] 导入 PromptServer 失败，请确认插件位于 custom_nodes 目录", exc_info=True)
    except Exception as e:
        logger.error(f"[CommonExtension v{__version__}] 路由注册总入口异常：{str(e)}", exc_info=True)

# ===================== 插件启动执行 =====================
# 非主模块时执行注册（ComfyUI 加载插件时）
if __name__ != "__main__":
    register_routes()

# ===================== 导出变量（ComfyUI 规范） =====================
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# 启动日志
logger.info(f"[CommonExtension v{__version__}] 插件主入口加载完成")