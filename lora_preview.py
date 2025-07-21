from aiohttp import web
import os
import folder_paths

class LoraPreviewPlugin:
    @classmethod
    def register_routes(cls, server):
        """
        注册 LoRA 预览路由
        :param server: PromptServer 实例（包含 app 属性）
        """
        # 1. 验证服务器实例
        if not hasattr(server, "app"):
            print("[LoraPreviewPlugin] 错误：服务器实例无 app 属性")
            return
        
        # 2. 确认 LoRA 目录路径（根据实际存放位置修改）
        # 注意：ComfyUI 默认目录是 "lora"，若你的是 "loras" 请修改
        lora_dir = os.path.join(folder_paths.models_dir, "loras")
        if not os.path.exists(lora_dir):
            print(f"[LoraPreviewPlugin] 错误：LoRA 目录不存在 - {lora_dir}")
            return
        
        # 3. 注册静态路由
        server.app.router.add_static(
            "/lora_previews",  # 访问 URL：http://127.0.0.1:8188/lora_previews/xxx.png
            lora_dir,          # 本地目录：ComfyUI/models/lora
            show_index=False   # 禁用目录索引（安全）
        )
        print(f"[LoraPreviewPlugin] 路由注册成功：/lora_previews -> {lora_dir}")

# 对外暴露的注册函数（供 __init__.py 调用）
def setup_plugin(server):
    LoraPreviewPlugin.register_routes(server)