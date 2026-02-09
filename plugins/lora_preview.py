import logging
import os
from aiohttp import web
import folder_paths
import comfy.samplers

# ===================== 全局配置 & 日志 =====================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ===================== Lora预览插件 =====================
class LoraPreviewPlugin:
    @classmethod
    def register_routes(cls, server):
        """完全恢复原始逻辑：注册静态路由，无新增接口方法"""
        # 1. 保留原有：检查路由表冻结逻辑
        if server.app.router.frozen:
            logger.info("[LoraPreviewPlugin] 路由表已冻结（热重载），跳过注册")
            return

        # 2. 保留原有：检查lora目录是否存在
        lora_dir = os.path.join(folder_paths.models_dir, "loras")
        if not os.path.exists(lora_dir):
            logger.warning(f"[LoraPreviewPlugin] 错误：目录不存在 - {lora_dir}")
            return

        try:
            # 3. 完全恢复原始逻辑：注册静态路由（无任何新增接口）
            #    客户端访问 /lora_previews/xxx.png 会直接映射到 lora_dir/xxx.png
            server.app.router.add_static(
                "/lora_previews",
                lora_dir,
                show_index=False  # 保留原有配置
            )
            logger.info(f"[LoraPreviewPlugin] 路由注册成功：/lora_previews -> {lora_dir}")
        except Exception as e:
            logger.error(f"[LoraPreviewPlugin] 注册失败：{e}")