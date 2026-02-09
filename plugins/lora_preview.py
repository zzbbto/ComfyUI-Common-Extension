import logging
import os
from aiohttp import web
import folder_paths

# ===================== 配置 & 日志 =====================
logger = logging.getLogger(__name__)

class LoraPreviewPlugin:
    @classmethod
    def register_routes(cls, server):
        """注册 /lora_previews GET 接口（仅获取Lora预览图片）"""
        # 1. 检查路由表是否冻结
        if server.app.router.frozen:
            logger.info(f"[LoraPreviewPlugin] 路由表已冻结（热重载），跳过注册")
            return

        try:
            # 2. 仅注册 /lora_previews 一个 GET 接口
            server.app.router.add_get("/lora_previews", cls.get_lora_preview)
            logger.info(f"[LoraPreviewPlugin] API 路由注册成功：/lora_previews")
        except Exception as e:
            logger.error(f"[LoraPreviewPlugin] 路由注册失败：{str(e)}", exc_info=True)

    @classmethod
    async def get_lora_previews(cls, request):
        """极简获取Lora预览图片（仅核心逻辑）"""
        try:
            # 1. 获取图片文件名参数
            filename = request.query.get("filename")
            if not filename:
                return web.json_response(
                    {"code": 400, "msg": "参数缺失：filename", "data": None},
                    status=400,
                    dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2)
                )

            # 2. 拼接Lora预览图片路径（核心：仅获取图片文件）
            lora_dir = os.path.join(folder_paths.models_dir, "loras")
            img_path = os.path.join(lora_dir, filename)

            # 3. 仅检查图片是否存在
            if not os.path.exists(img_path):
                return web.json_response(
                    {"code": 404, "msg": "预览图片不存在", "data": None},
                    status=404,
                    dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2)
                )

            # 4. 直接返回图片文件流（无任何多余逻辑）
            return web.FileResponse(path=img_path)
        
        except Exception as e:
            logger.error(f"[LoraPreviewPlugin] 获取预览图片失败：{str(e)}", exc_info=True)
            return web.json_response(
                {"code": 500, "msg": "获取图片失败", "data": None},
                status=500,
                dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2)
            )