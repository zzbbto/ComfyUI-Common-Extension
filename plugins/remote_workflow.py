import json
import logging
import os
import stat
from aiohttp import web
import folder_paths

# ===================== 配置 & 日志 =====================
logger = logging.getLogger(__name__)

# === 适配ComfyUI-Manager 固定工作流路径 ===
USER_WORKFLOW_DIR = os.path.join(folder_paths.base_path, "user", "default", "workflows")
os.makedirs(USER_WORKFLOW_DIR, exist_ok=True)

# 导入 PromptServer（和第一种完全一致）
try:
    from server import PromptServer
except ImportError as e:
    logger.error("Could not import PromptServer. Make sure this is installed in ComfyUI's custom_nodes directory.")
    raise e

# ===================== 工具方法（不变） =====================
def _safe_join(base_dir, filename):
    try:
        abs_base = os.path.abspath(base_dir)
        abs_path = os.path.abspath(os.path.join(base_dir, filename))
        if abs_path.startswith(abs_base) and abs_path.endswith(".json"):
            return abs_path
        else:
            logger.warning(f"[RemoteWorkflowPlugin] 非法路径访问：{filename}（超出基础目录）")
            return None
    except Exception as e:
        logger.error(f"[RemoteWorkflowPlugin] 路径校验失败：{filename} - {str(e)}")
        return None

# ===================== 路由：列出所有工作流（装饰器注册，同第一种风格） =====================
@PromptServer.instance.routes.get("/api/remote_wf/list")
async def list_all_user_workflows(request):
    try:
        workflows = []
        for root, dirs, files in os.walk(USER_WORKFLOW_DIR):
            for filename in files:
                if not filename.lower().endswith(".json") or filename.startswith("."):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    file_stat = os.stat(file_path)
                    rel_filename = os.path.relpath(file_path, USER_WORKFLOW_DIR)
                    workflows.append({
                        "filename": rel_filename,
                        "size": file_stat.st_size,
                        "mtime": file_stat.st_mtime
                    })
                except PermissionError:
                    logger.warning(f"[RemoteWorkflowPlugin] 无权限访问文件：{file_path}")
                    continue
                except Exception as e:
                    logger.error(f"[RemoteWorkflowPlugin] 读取文件信息失败：{file_path} - {str(e)}")
                    continue

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

# ===================== 路由：获取单个工作流（装饰器注册，同第一种风格） =====================
@PromptServer.instance.routes.get("/api/remote_wf/get")
async def get_single_user_workflow(request):
    try:
        filename = request.query.get("filename")
        if not filename:
            return web.json_response({
                "code": 400,
                "msg": "参数缺失：filename",
                "data": None,
            }, status=400, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))

        file_path = _safe_join(USER_WORKFLOW_DIR, filename)
        if not file_path:
            return web.json_response({
                "code": 403,
                "msg": "非法文件名：禁止访问基础目录外的文件",
                "filename": filename,
                "data": None,
            }, status=403, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))

        if not os.path.exists(file_path):
            return web.json_response({
                "code": 404,
                "msg": "工作流文件不存在",
                "filename": filename,
                "data": None,
            }, status=404, dumps=lambda x: web.json_dumps(x, ensure_ascii=False, indent=2))

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

# 注册成功日志（和第一种一致）
logger.info("Remote Workflow API endpoints registered: /api/remote_wf/list, /api/remote_wf/get")