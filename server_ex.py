import json
import traceback
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import uvicorn
from typing import Dict, Any, List, Tuple, Union

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ComfyUI Workflow Converter API",
    description="将ComfyUI工作流转换为API格式",
    version="1.0.0"
)

@app.post("/convert", summary="转换工作流")
async def convert_workflow(request: Request):
    """
    将上传的ComfyUI工作流JSON字符串转换为API格式
    - **请求体**: ComfyUI工作流JSON字符串
    """
    try:
        # 读取请求体中的JSON字符串
        body = await request.body()
        workflow_str = body.decode("utf-8")
        workflow = json.loads(workflow_str)
        
        # 转换工作流
        api_workflow = convert_comfyui_workflow(workflow)
        
        # 返回转换后的JSON字符串
        return JSONResponse(content=api_workflow)
    except json.JSONDecodeError:
        raise HTTPException(400, "无效的JSON格式")
    except Exception as e:
        logger.error(f"转换失败: {str(e)}", exc_info=True)
        raise HTTPException(500, f"转换失败: {str(e)}")

@app.post("/convert-to-file", summary="转换工作流并返回文件")
async def convert_to_file(request: Request):
    """
    将上传的ComfyUI工作流JSON字符串转换为API格式并返回JSON文件
    - **请求体**: ComfyUI工作流JSON字符串
    """
    try:
        # 读取请求体中的JSON字符串
        body = await request.body()
        workflow_str = body.decode("utf-8")
        workflow = json.loads(workflow_str)
        
        # 转换工作流
        api_workflow = convert_comfyui_workflow(workflow)
        
        # 创建临时文件保存结果
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as tmp:
            json.dump(api_workflow, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
        
        # 返回文件下载
        return FileResponse(
            tmp_path,
            media_type='application/json',
            filename="api_workflow.json"
        )
    except json.JSONDecodeError:
        raise HTTPException(400, "无效的JSON格式")
    except Exception as e:
        logger.error(f"转换失败: {str(e)}", exc_info=True)
        raise HTTPException(500, f"转换失败: {str(e)}")

def convert_comfyui_workflow(workflow: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    将ComfyUI工作流转换为API格式，处理虚拟链接和UI字段过滤
    
    参数:
        workflow (dict): ComfyUI工作流JSON对象
        
    返回:
        dict: API格式的工作流
    """
    # 1. 处理链接格式（支持数组格式和对象格式）
    existing_links = []
    raw_links = workflow.get("links", [])
    
    # 将数组格式的链接转换为对象格式
    for link in raw_links:
        if isinstance(link, list) and len(link) >= 6:
            # 数组格式: [id, from_node, from_slot, to_node, to_slot, type]
            link_obj = {
                "id": link[0],
                "from_node": link[1],
                "from_slot": link[2],
                "to_node": link[3],
                "to_slot": link[4],
                "type": link[5] if len(link) > 5 else ""
            }
            existing_links.append(link_obj)
        elif isinstance(link, dict):
            # 已经是对象格式
            existing_links.append(link)
    
    # 2. 处理"Use Everywhere"虚拟链接
    extra_data = workflow.get("extra", {})
    ue_links = []
    if isinstance(extra_data, list):
        # 如果extra是列表，尝试找到包含ue_links的字典
        for item in extra_data:
            if isinstance(item, dict):
                ue_links.extend(item.get("ue_links", []))
    else:
        # 如果extra是字典
        ue_links = extra_data.get("ue_links", [])
    
    # 创建链接ID映射表 (link_id -> (from_node, from_slot))
    link_index = {}
    max_link_id = max((link.get("id", 0) for link in existing_links), default=0)
    for link in existing_links:
        link_id = link["id"]
        link_index[link_id] = (link["from_node"], link["from_slot"])
    
    # 添加虚拟链接
    for ue_link in ue_links:
        max_link_id += 1
        new_link = {
            "id": max_link_id,
            "from_node": int(ue_link["upstream"]),
            "from_slot": ue_link["upstream_slot"],
            "to_node": int(ue_link["downstream"]),
            "to_slot": ue_link["downstream_slot"],
            "type": ue_link["type"]
        }
        existing_links.append(new_link)
        link_index[max_link_id] = (new_link["from_node"], new_link["from_slot"])
    
    # 3. 转换节点格式
    api_workflow = {}
    node_map = {node["id"]: node for node in workflow["nodes"]}
    
    for node in workflow["nodes"]:
        node_id = str(node["id"])
        node_type = node["type"]
        
        # 跳过UI特定节点
        if node_type in ["Reroute", "Note", "Group"]:
            continue
        
        # 创建API节点
        api_node = {
            "inputs": {},
            "class_type": node_type,
            "_meta": {
                "title": node.get("properties", {}).get("Node name for S&R", node_type)
            }
        }
        
        # 分离处理控件参数和链接参数
        control_inputs = []  # 控件参数（带widget）
        link_inputs = {}     # 链接参数
        
        # 第一步：收集所有输入参数信息
        if "inputs" in node:
            for input_slot in node["inputs"]:
                input_name = input_slot["name"]
                # 处理链接输入（优先处理链接）
                if "link" in input_slot and input_slot["link"] is not None:
                    link_id = input_slot["link"]
                    if link_id in link_index:
                        from_node, from_slot = link_index[link_id]
                        # 验证源节点存在
                        if from_node in node_map:
                            link_inputs[input_name] = [str(from_node), from_slot]
                # 收集控件输入（带widget的输入）
                if "widget" in input_slot:
                    control_inputs.append(input_name)
        
        # 第二步：处理控件值(widgets_values)
        widgets_values = node.get("widgets_values", [])
        
        # 关键修复：确保widgets_values是列表类型
        if not isinstance(widgets_values, list):
            logger.warning(f"节点 {node_id} (类型: {node_type}) 的widgets_values不是列表类型，已转换为空列表")
            widgets_values = []
        
        # 处理控件值映射
        for i in range(len(widgets_values)):
            try:
                # 获取输入名称
                if i < len(control_inputs):
                    input_name = control_inputs[i]
                else:
                    input_name = f"widget_{i}"
                
                # 特殊节点处理
                if node_type == "easy stylesSelector":
                    if i == 0:
                        input_name = "styles"
                    elif i == 1:
                        input_name = "select_styles"
                
                # 获取值并进行类型转换
                value = widgets_values[i]
                
                # 针对特定节点进行类型转换
                if node_type == "KSampler":
                    if input_name == "steps":
                        try:
                            value = int(value)
                        except (ValueError, TypeError):
                            pass  # 保留原始值
                    elif input_name in ["cfg", "denoise"]:
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            pass  # 保留原始值
                
                api_node["inputs"][input_name] = value
            except IndexError:
                logger.warning(f"节点 {node_id} (类型: {node_type}) 缺少widget值 at index {i}")
            except Exception as e:
                logger.error(f"处理节点 {node_id} 的widget值时出错: {str(e)}", exc_info=True)
        
        # 第三步：添加链接参数（放在控件参数之后，保证顺序一致）
        for input_name, value in link_inputs.items():
            api_node["inputs"][input_name] = value
        
        api_workflow[node_id] = api_node
    
    return api_workflow

def run_server(host="127.0.0.1", port=8000):
    """运行HTTP服务器"""
    print(f"🚀 ComfyUI工作流转换服务启动于 http://{host}:{port}")
    print(f"🔁 转换端点: POST /convert")
    print(f"📥 文件下载端点: POST /convert-to-file")
    print("🛑 按 Ctrl+C 停止服务")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()