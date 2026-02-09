import logging
import os
from aiohttp import web
import folder_paths

# ===================== 配置 & 日志 =====================
logger = logging.getLogger(__name__)

# 安全配置
MAX_STATIC_FILE_SIZE = 50 * 1024 * 1024  # 静态文件最大50MB
CACHE_CONTROL = "max-age=3600"  # 缓存1小时

class LoraPreviewPlugin:
    @classmethod
    def register_routes(cls, server):
        """注册 Lora 预览静态文件路由"""
        # 1. 检查路由表是否冻结（热重载场景）
        if server.app.router.frozen:
            logger.info(f"[LoraPreviewPlugin] 路由表已冻结（热重载），跳过注册")
            return

        # 2. 验证 Lora 目录
        lora_dir = os.path.join(folder_paths.models_dir, "loras")
        if not os.path.exists(lora_dir):
            logger.error(f"[LoraPreviewPlugin] Lora 目录不存在：{lora_dir}")
            return

        try:
            # 3. 注册静态路由（增强安全配置）
            server.app.router.add_static(
                "/lora_previews",
                lora_dir,
                show_index=False,        # 禁止目录索引
                max_size=MAX_STATIC_FILE_SIZE,  # 文件大小限制
                cache_control=CACHE_CONTROL,    # 缓存控制
                follow_symlinks=False    # 禁止软链接（安全）
            )
            logger.info(f"[LoraPreviewPlugin] 静态路由注册成功：/lora_previews -> {lora_dir}")
        except Exception as e:
            logger.error(f"[LoraPreviewPlugin] 路由注册失败：{str(e)}", exc_info=True)