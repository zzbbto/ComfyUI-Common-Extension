import logging
from aiohttp import web
import comfy.samplers

# ===================== 配置 & 日志 =====================
logger = logging.getLogger(__name__)

# 导入 PromptServer（统一注册风格）
try:
    from server import PromptServer
except ImportError as e:
    logger.error("Could not import PromptServer. Make sure this is installed in ComfyUI's custom_nodes directory.")
    raise e

# ===================== 路由：获取采样器列表 =====================
@PromptServer.instance.routes.get("/api/sampler")
async def get_samplers(request):
    """获取所有采样器列表（标准化响应）"""
    try:
        # 核心逻辑完全保留
        samplers = comfy.samplers.KSampler.SAMPLERS
        return web.json_response(
            {
                "code": 200,
                "msg": "success",
                "data": samplers,
            },
            dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2)
        )
    except Exception as e:
        logger.error(f"[SamplerSchedulerPlugin] 获取采样器失败：{str(e)}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)

# ===================== 路由：获取调度器列表 =====================
@PromptServer.instance.routes.get("/api/scheduler")
async def get_schedulers(request):
    """获取所有调度器列表（标准化响应）"""
    try:
        # 核心逻辑完全保留
        schedulers = comfy.samplers.KSampler.SCHEDULERS
        return web.json_response(
            {
                "code": 200,
                "msg": "success",
                "data": schedulers,
            },
            dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2)
        )
    except Exception as e:
        logger.error(f"[SamplerSchedulerPlugin] 获取调度器失败：{str(e)}", exc_info=True)
        return web.json_response({"error": str(e)}, status=500)

# 注册成功日志
logger.info("[SamplerSchedulerPlugin] 接口注册成功：/api/sampler | /api/scheduler")