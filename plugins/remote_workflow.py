"""
RemoteWorkflowPlugin - 适配ComfyUI-Manager的远程工作流API
提供：列出用户工作流、获取单个工作流、列出模板工作流
对齐整体插件风格（日志、版本、热重载、标准化响应）
Version: 1.0.0
"""
import json
import logging
import os
import stat
from aiohttp import web
import folder_paths
import server

# ===================== 配置 & 日志 =====================
logger = logging.getLogger(__name__)

# === 适配ComfyUI-Manager 固定工作流路径 ===
# 用户手动保存的工作流（核心）
USER_WORKFLOW_DIR = os.path.join(folder_paths.base_path, "user", "default", "workflows")

# 确保目录存在，避免首次运行报错
os.makedirs(USER_WORKFLOW_DIR, exist_ok=True)

class RemoteWorkflowPlugin:
    @classmethod
    def register_routes(cls, server_instance):
        """注册远程工作流所有API路由"""
        # 1. 检查路由表是否冻结（热重载场景）
        if server_instance.app.router.frozen:
            logger.info(f"[RemoteWorkflowPlugin] 路由表已冻结（热重载），跳过注册")
            return

        try:
            # 2. 注册所有远程工作流API路由
            server_instance.app.router.add_get("/api/remote_wf/list", cls.list_all_user_workflows)
            server_instance.app.router.add_get("/api/remote_wf/get", cls.get_single_user_workflow)
            logger.info(f"[RemoteWorkflowPlugin] 远程工作流API路由注册成功")
        except Exception as e:
            logger.error(f"[RemoteWorkflowPlugin] 路由注册失败：{str(e)}", exc_info=True)

    # === 公开API 1：列出所有用户保存的工作流 ===
    @classmethod
    async def list_all_user_workflows(cls, request):
        try:
            workflows = []
            # 遍历目录，增强容错（处理目录访问异常）
            for filename in os.listdir(USER_WORKFLOW_DIR):
                # 过滤非JSON文件和隐藏文件
                if not filename.lower().endswith(".json") or filename.startswith("."):
                    continue
                
                file_path = os.path.join(USER_WORKFLOW_DIR, filename)
                try:
                    # 获取文件信息（处理权限问题）
                    file_stat = os.stat(file_path)
                    workflows.append({
                        "filename": filename,
                        "size": file_stat.st_size,  # 文件大小（字节）
                        "mtime": file_stat.st_mtime  # 最后修改时间（时间戳）
                    })
                except PermissionError:
                    logger.warning(f"[RemoteWorkflowPlugin] 无权限访问文件：{file_path}")
                    continue

            # 标准化响应
            return web.json_response({
                "code": 200,
                "msg": "success",
                "count": len(workflows),
                "data": workflows,
                "base_dir": USER_WORKFLOW_DIR,
            }, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))
        
        except Exception as e:
            logger.error(f"[RemoteWorkflowPlugin] 列出用户工作流失败：{str(e)}", exc_info=True)
            return web.json_response({
                "code": 500,
                "msg": f"列出用户工作流失败：{str(e)}",
                "count": 0,
                "data": None,
                "base_dir": USER_WORKFLOW_DIR,
            }, status=500, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))

    # === 公开API 2：获取单个用户工作流原始JSON ===
    @classmethod
    async def get_single_user_workflow(cls, request):
        try:
            # 1. 获取并校验参数
            filename = request.query.get("filename")
            if not filename:
                return web.json_response({
                    "code": 400,
                    "msg": "参数缺失：filename",
                    "data": None,
                }, status=400, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))
            
            # 2. 校验文件格式
            if not filename.lower().endswith(".json"):
                return web.json_response({
                    "code": 400,
                    "msg": "文件格式错误：仅支持.json文件",
                    "data": None,
                }, status=400, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))
            
            # 3. 校验文件存在性
            file_path = os.path.join(USER_WORKFLOW_DIR, filename)
            if not os.path.exists(file_path):
                return web.json_response({
                    "code": 404,
                    "msg": "工作流文件不存在",
                    "filename": filename,
                    "data": None,
                }, status=404, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))
            
            # 4. 读取并返回工作流
            with open(file_path, "r", encoding="utf-8") as f:
                workflow_json = json.load(f)
            
            return web.json_response({
                "code": 200,
                "msg": "success",
                "filename": filename,
                "data": workflow_json,
            }, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))
        
        except json.JSONDecodeError as e:
            logger.error(f"[RemoteWorkflowPlugin] 工作流文件JSON解析失败：{filename} - {str(e)}", exc_info=True)
            return web.json_response({
                "code": 500,
                "msg": f"工作流文件JSON解析失败：{str(e)}",
                "filename": filename,
                "data": None,
            }, status=500, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))
        
        except PermissionError:
            logger.error(f"[RemoteWorkflowPlugin] 无权限读取文件：{filename}", exc_info=True)
            return web.json_response({
                "code": 500,
                "msg": "无权限读取工作流文件",
                "filename": filename,
                "data": None,
            }, status=500, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))
        
        except Exception as e:
            logger.error(f"[RemoteWorkflowPlugin] 获取单个工作流失败：{str(e)}", exc_info=True)
            return web.json_response({
                "code": 500,
                "msg": f"服务器内部错误：{str(e)}",
                "filename": request.query.get("filename"),
                "data": None,
            }, status=500, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))

# 启动日志（首次加载时输出）
logger.info(f"[RemoteWorkflowPlugin v{__version__}] 模块加载完成，等待路由注册")