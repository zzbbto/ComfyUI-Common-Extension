import logging
from aiohttp import web
import comfy.samplers

# ===================== 配置 & 日志 =====================
logger = logging.getLogger(__name__)

class SamplerSchedulerPlugin:
    @classmethod
    def register_routes(cls, server):
        """注册 Sampler/Scheduler API 路由"""
        # 1. 检查路由表是否冻结
        if server.app.router.frozen:
            logger.info(f"[SamplerSchedulerPlugin] 路由表已冻结（热重载），跳过注册")
            return

        try:
            # 2. 注册 API 路由
            server.app.router.add_get("/api/sampler", cls.get_samplers)
            server.app.router.add_get("/api/scheduler", cls.get_schedulers)
            logger.info(f"[SamplerSchedulerPlugin] API 路由注册成功：/api/sampler | /api/scheduler")
        except Exception as e:
            logger.error(f"[SamplerSchedulerPlugin] 路由注册失败：{str(e)}", exc_info=True)

    @classmethod
    async def get_samplers(cls, request):
        """获取所有采样器列表（标准化响应）"""
        try:
            samplers = comfy.samplers.KSampler.SAMPLERS
            # 对齐 workflow-to-api-converter 的响应格式（带缩进、UTF8）
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
            return web.json_response(
                {
                    "code": 500,
                    "msg": f"获取采样器失败：{str(e)}",
                    "data": None,
                },
                status=500,
                dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2)
            )

    @classmethod
    async def get_schedulers(cls, request):
        """获取所有调度器列表（标准化响应）"""
        try:
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
            return web.json_response(
                {
                    "code": 500,
                    "msg": f"获取调度器失败：{str(e)}",
                    "data": None,
                },
                status=500,
                dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2)
            )