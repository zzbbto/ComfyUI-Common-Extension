import json
import logging
import os
import stat
import ssl
from urllib.parse import urlparse, urljoin
from aiohttp import web, ClientSession, ClientError
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

# ===================== 路由：列出所有工作流（修复json_dumps错误） =====================
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
                    # 相对路径（如 test/api/test.json）
                    rel_filename = os.path.relpath(file_path, USER_WORKFLOW_DIR)
                    # 统一路径分隔符为 /（兼容Windows）
                    rel_filename = rel_filename.replace("\\", "/")
                    
                    # 第一步：获取完整父目录（如 test/api/test.json → test/api）
                    full_parent_dir = os.path.dirname(rel_filename)
                    # 第二步：仅提取第一级目录
                    if full_parent_dir == "":
                        # 根目录文件，父目录为空
                        parent_dir = ""
                    else:
                        # 分割路径，取第一级（如 test/api → test，api → api）
                        parent_dir = full_parent_dir.split("/")[0]
                    
                    workflows.append({
                        "filename": rel_filename,  # 完整相对路径（如 test/api/test.json）
                        "parent_dir": parent_dir,  # 仅第一级父目录（如 test）
                        "size": file_stat.st_size, # 文件大小（字节）
                        "mtime": file_stat.st_mtime # 修改时间（时间戳）
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
        }, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复：web.json_dumps → json.dumps

    except Exception as e:
        logger.error(f"[RemoteWorkflowPlugin] 列出用户工作流失败：{str(e)}", exc_info=True)
        return web.json_response({
            "code": 500,
            "msg": f"列出用户工作流失败：{str(e)}",
            "count": 0,
            "data": None,
            "base_dir": USER_WORKFLOW_DIR,
        }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复：web.json_dumps → json.dumps

# ===================== 路由：获取单个工作流（修复json_dumps错误） =====================
@PromptServer.instance.routes.get("/api/remote_wf/get")
async def get_single_user_workflow(request):
    try:
        filename = request.query.get("filename")
        if not filename:
            return web.json_response({
                "code": 400,
                "msg": "参数缺失：filename",
                "data": None,
            }, status=400, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

        file_path = _safe_join(USER_WORKFLOW_DIR, filename)
        if not file_path:
            return web.json_response({
                "code": 403,
                "msg": "非法文件名：禁止访问基础目录外的文件",
                "filename": filename,
                "data": None,
            }, status=403, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

        if not os.path.exists(file_path):
            return web.json_response({
                "code": 404,
                "msg": "工作流文件不存在",
                "filename": filename,
                "data": None,
            }, status=404, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

        with open(file_path, "r", encoding="utf-8") as f:
            workflow_json = json.load(f)

        return web.json_response({
            "code": 200,
            "msg": "success",
            "filename": filename,
            "data": workflow_json,
        }, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

    except json.JSONDecodeError as e:
        logger.error(f"[RemoteWorkflowPlugin] 工作流文件JSON解析失败：{filename} - {str(e)}", exc_info=True)
        return web.json_response({
            "code": 500,
            "msg": f"工作流文件JSON解析失败：{str(e)}",
            "filename": filename,
            "data": None,
        }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

    except PermissionError:
        logger.error(f"[RemoteWorkflowPlugin] 无权限读取文件：{filename}", exc_info=True)
        return web.json_response({
            "code": 500,
            "msg": "无权限读取工作流文件",
            "filename": filename,
            "data": None,
        }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

    except Exception as e:
        logger.error(f"[RemoteWorkflowPlugin] 获取单个工作流失败：{str(e)}", exc_info=True)
        return web.json_response({
            "code": 500,
            "msg": f"服务器内部错误：{str(e)}",
            "filename": request.query.get("filename"),
            "data": None,
        }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

# ===================== 核心路由：获取API格式工作流（修复json_dumps错误） =====================
@PromptServer.instance.routes.get("/api/remote_wf/get_api")
async def get_single_workflow_api_format(request):
    """
    获取指定工作流文件的API格式JSON
    参数：
        filename - 必选，工作流文件名（如 "test_workflow.json"）
        convert_host - 可选，转换接口基础地址（仅HTTP，如 http://127.0.0.1:8188），不传则自动适配当前ComfyUI地址
    """
    try:
        # 1. 校验核心参数
        filename = request.query.get("filename")
        if not filename:
            return web.json_response({
                "code": 400,
                "msg": "参数缺失：filename",
                "data": None,
            }, status=400, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

        # 2. 安全校验文件路径
        file_path = _safe_join(USER_WORKFLOW_DIR, filename)
        if not file_path:
            return web.json_response({
                "code": 403,
                "msg": "非法文件名：禁止访问基础目录外的文件",
                "filename": filename,
                "data": None,
            }, status=403, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

        # 3. 校验文件是否存在
        if not os.path.exists(file_path):
            return web.json_response({
                "code": 404,
                "msg": "工作流文件不存在",
                "filename": filename,
                "data": None,
            }, status=404, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

        # 4. 读取原始工作流
        with open(file_path, "r", encoding="utf-8") as f:
            workflow_json = json.load(f)
        logger.info(f"[RemoteWorkflowPlugin] 读取工作流文件：{filename}")

        # 5. 处理转换地址（仅接收基础IP地址，自动拼接路径）
        convert_host = request.query.get("convert_host")
        convert_url = None
        ssl_context = None  # 仅HTTP，无需SSL

        # 5.1 传入了convert_host（如 http://127.0.0.1:8188）
        if convert_host:
            # 自动拼接 /workflow/convert 路径（兼容结尾带/的情况）
            convert_url = urljoin(convert_host, "/workflow/convert")
            logger.info(f"[RemoteWorkflowPlugin] 使用外部基础地址：{convert_host} → 拼接后：{convert_url}")

        # 5.2 未传入则自动适配当前ComfyUI地址（默认 http://127.0.0.1:8188）
        if not convert_url:
            server_addr = PromptServer.instance.address or "127.0.0.1"
            server_port = PromptServer.instance.port or 8188
            convert_url = f"http://{server_addr}:{server_port}/workflow/convert"
            logger.info(f"[RemoteWorkflowPlugin] 使用自动适配地址：{convert_url}")

        # 6. 调用转换接口（仅HTTP，简化逻辑）
        async with ClientSession(ssl=ssl_context) as session:
            async with session.post(
                convert_url,
                json=workflow_json,
                timeout=30
            ) as resp:
                if resp.status != 200:
                    error_data = await resp.json() if resp.content_type == "application/json" else await resp.text()
                    logger.error(f"转换接口调用失败：{error_data}")
                    return web.json_response({
                        "code": 500,
                        "msg": f"工作流转换失败：{error_data}",
                        "filename": filename,
                        "data": None,
                    }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

                api_format_json = await resp.json()

        # 7. 返回结果
        return web.json_response({
            "code": 200,
            "msg": "success",
            "filename": filename,
            "convert_url_used": convert_url,
            "data": api_format_json,
        }, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

    # 异常处理（简化，适配本地HTTP场景）
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败：{filename} - {str(e)}")
        return web.json_response({
            "code": 500,
            "msg": f"JSON解析失败：{str(e)}",
            "filename": filename,
            "data": None,
        }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

    except ClientError as e:
        logger.error(f"接口调用失败：{str(e)}")
        return web.json_response({
            "code": 500,
            "msg": f"转换接口调用失败：{str(e)}（请确认地址可访问）",
            "filename": filename,
            "data": None,
        }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

    except PermissionError:
        logger.error(f"无权限读取文件：{filename}")
        return web.json_response({
            "code": 500,
            "msg": "无权限读取工作流文件",
            "filename": filename,
            "data": None,
        }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

    except Exception as e:
        logger.error(f"内部错误：{str(e)}", exc_info=True)
        return web.json_response({
            "code": 500,
            "msg": f"内部错误：{str(e)}",
            "filename": filename,
            "data": None,
        }, status=500, dumps=lambda x: json.dumps(x, ensure_ascii=False, indent=2))  # 修复

# 注册日志
logger.info("Remote Workflow API endpoints registered: /api/remote_wf/list, /api/remote_wf/get, /api/remote_wf/get_api")