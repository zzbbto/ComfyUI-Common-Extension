from aiohttp import web
import os
import folder_paths
import comfy.samplers

class LoraPreviewPlugin:
    @classmethod
    def register_routes(cls, server):
        # 关键：检查路由表是否冻结（热重载时为 True）
        if server.app.router.frozen:
            print("[LoraPreviewPlugin] 路由表已冻结（热重载），跳过注册")
            return

        lora_dir = os.path.join(folder_paths.models_dir, "loras")
        if not os.path.exists(lora_dir):
            print(f"[LoraPreviewPlugin] 错误：目录不存在 - {lora_dir}")
            return

        try:
            server.app.router.add_static(
                "/lora_previews",
                lora_dir,
                show_index=False
            )
            print(f"[LoraPreviewPlugin] 路由注册成功：/lora_previews -> {lora_dir}")
        except Exception as e:
            print(f"[LoraPreviewPlugin] 注册失败：{e}")

class SamplerSchedulerPlugin:
    @classmethod
    def register_routes(cls, server):
        # 关键：检查路由表是否冻结（热重载时为 True）
        if server.app.router.frozen:
            print("[SamplerSchedulerPlugin] 路由表已冻结（热重载），跳过注册")
            return

        try:
            server.app.router.add_get("/api/sampler", cls.get_samplers)
            server.app.router.add_get("/api/scheduler", cls.get_schedulers)
            print("[SamplerSchedulerPlugin] 接口注册成功")
        except Exception as e:
            print(f"[SamplerSchedulerPlugin] 注册失败：{e}")

    @classmethod
    async def get_samplers(cls, request):
        try:
            return web.json_response(comfy.samplers.KSampler.SAMPLERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @classmethod
    async def get_schedulers(cls, request):
        try:
            return web.json_response(comfy.samplers.KSampler.SCHEDULERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

def setup_plugin(server):
    LoraPreviewPlugin.register_routes(server)
    SamplerSchedulerPlugin.register_routes(server)