import importlib
import logging
import os
import sys
from pathlib import Path

# ===================== 基础配置 =====================
logger = logging.getLogger(__name__)
__version__ = "1.0.0"

# 获取插件根目录
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIRECTORY)

# ===================== 节点注册 =====================
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 导入 scripts_common 下的节点（保留原有逻辑）
scripts_dir = os.path.join(SCRIPT_DIRECTORY, "scripts_common")
if os.path.exists(scripts_dir):
    for py_file in Path(scripts_dir).rglob("*.py"):
        if py_file.name.startswith("__") or "__pycache__" in py_file.parts:
            continue
        rel_path = py_file.relative_to(SCRIPT_DIRECTORY)
        module_path = str(rel_path.with_suffix("")).replace(os.sep, ".")
        try:
            mod = importlib.import_module(module_path)
            NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
            NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        except Exception as e:
            logger.error(f"[CommonExtension v{__version__}] 导入模块 {module_path} 失败：{str(e)}", exc_info=True)
else:
    logger.warning(f"[CommonExtension v{__version__}] 未找到 scripts_common 目录：{scripts_dir}")

# ===================== 路由注册总入口（混合风格：手动+自动） =====================
def register_routes():
    """适配混合注册风格：LoraPreview手动注册，其他插件自动注册"""
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

        # 1. 手动注册 LoraPreviewPlugin（保留原有逻辑）
        try:
            from .plugins.lora_preview import LoraPreviewPlugin
            LoraPreviewPlugin.register_routes(server_instance)
            logger.info(f"[CommonExtension v{__version__}] LoraPreviewPlugin 路由手动注册完成")
        except ImportError as e:
            logger.error(f"[CommonExtension v{__version__}] 导入 LoraPreviewPlugin 失败：{str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"[CommonExtension v{__version__}] LoraPreviewPlugin 注册失败：{str(e)}", exc_info=True)

        # 2. 导入 SamplerScheduler/RemoteWorkflow 触发装饰器自动注册
        try:
            from .plugins import sampler_scheduler
            from .plugins import remote_workflow
        except ImportError as e:
            logger.error(f"[CommonExtension v{__version__}] 加载自动注册插件失败：{str(e)}", exc_info=True)

        logger.info(f"[CommonExtension v{__version__}] 所有插件路由注册完成（混合风格）")
    except ImportError as e:
        logger.error(f"[CommonExtension v{__version__}] 导入 PromptServer 失败，请确认插件位于 custom_nodes 目录", exc_info=True)
    except Exception as e:
        logger.error(f"[CommonExtension v{__version__}] 路由注册总入口异常：{str(e)}", exc_info=True)

# ===================== 插件启动执行 =====================
if __name__ != "__main__":
    # 执行路由注册（手动+自动）
    register_routes()

# ===================== 导出变量（ComfyUI 规范） =====================
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# 启动日志
logger.info(f"[CommonExtension v{__version__}] 插件主入口加载完成（混合路由注册风格）")