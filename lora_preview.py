from aiohttp import web
import os
import folder_paths
import comfy.samplers  # 导入 ComfyUI 的采样器模块

class LoraPreviewPlugin:
    @classmethod
    def register_routes(cls, server):
        """注册 LoRA 预览静态路由"""
        # 验证服务器实例是否包含 app
        if not hasattr(server, "app"):
            print("[LoraPreviewPlugin] 错误：服务器实例无 app 属性")
            return
        
        # 配置 LoRA 目录路径（根据实际存放位置修改）
        lora_dir = os.path.join(folder_paths.models_dir, "loras")
        if not os.path.exists(lora_dir):
            print(f"[LoraPreviewPlugin] 错误：LoRA 目录不存在 - {lora_dir}")
            return
        
        # 注册静态路由
        server.app.router.add_static(
            "/lora_previews",  # 访问 URL：http://127.0.0.1:8188/lora_previews/xxx.png
            lora_dir,          # 本地目录：ComfyUI/models/loras
            show_index=False   # 禁用目录索引（安全）
        )
        print(f"[LoraPreviewPlugin] LoRA 路由注册成功：/lora_previews -> {lora_dir}")

class SamplerSchedulerPlugin:
    @classmethod
    def register_routes(cls, server):
        """注册采样器和调度器接口路由"""
        # 验证服务器实例是否包含 app
        if not hasattr(server, "app"):
            print("[SamplerSchedulerPlugin] 错误：服务器实例无 app 属性")
            return
        
        # 注册采样器列表接口
        server.app.router.add_get("/api/sampler", cls.get_samplers)
        # 注册调度器列表接口
        server.app.router.add_get("/api/scheduler", cls.get_schedulers)
        print("[SamplerSchedulerPlugin] 采样器/调度器接口注册成功")

    @classmethod
    async def get_samplers(cls, request):
        """返回 ComfyUI 支持的采样器列表"""
        try:
            # 从 ComfyUI 的 KSampler 中获取采样器列表
            samplers = comfy.samplers.KSampler.SAMPLERS
            return web.json_response(samplers)
        except Exception as e:
            print(f"[SamplerSchedulerPlugin] 采样器接口错误：{e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)

    @classmethod
    async def get_schedulers(cls, request):
        """返回 ComfyUI 支持的调度器列表"""
        try:
            # 从 ComfyUI 的 KSampler 中获取调度器列表
            schedulers = comfy.samplers.KSampler.SCHEDULERS
            return web.json_response(schedulers)
        except Exception as e:
            print(f"[SamplerSchedulerPlugin] 调度器接口错误：{e}")
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)

# 对外暴露的注册函数（供 __init__.py 调用，整合所有路由）
def setup_plugin(server):
    # 注册 LoRA 预览路由
    LoraPreviewPlugin.register_routes(server)
    # 注册采样器和调度器接口
    SamplerSchedulerPlugin.register_routes(server)