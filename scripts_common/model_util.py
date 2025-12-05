import gc
import math
from typing import Optional, Tuple, List, Dict, Any
import torch
import logging
from tqdm import tqdm
import comfy.model_management
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher
from comfy.model_base import WAN21, QwenImage, Flux
from memory_clear import get_system_vram_usage, get_total_gpu_memory
from memory_util import clean_ram

# ---------------------- 全局配置与日志初始化 ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [BlockSwap] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ModelBlockLogger")

# 共享常量
UNET_CANDIDATES: List[str] = ['unet', 'diffusion_model', 'model', 'denoise_fn', 'network', 'transformer_unet']
BLOCK_ATTR_CANDIDATES: List[str] = [
    'blocks', 'transformer_blocks', 'down', 'up', 'mid', 'layers', 
    'resblocks', 'single_blocks', 'double_blocks'
]
AUTO_INIT_PERCENT: int = 20
AUTO_SWAP_STEP: int = 20
DEFAULT_TXT_EMB_ATTR: str = "text_embedding"
DEFAULT_IMG_EMB_ATTR: str = "img_emb"
MIN_VR_AM_SINGLE: int = 40

# ---------------------- 核心工具函数（类型注解+逻辑优化） ----------------------
def is_cuda_device(device: Any) -> bool:
    """统一判断设备是否为CUDA"""
    if isinstance(device, str):
        return device.startswith('cuda')
    elif isinstance(device, torch.device):
        return device.type == 'cuda'
    return False

def safe_move_module(module: torch.nn.Module, target_device: torch.device, non_blocking: bool = False) -> bool:
    """
    安全迁移模块所有参数/缓冲区到目标设备（递归处理子模块）
    修复：仅CUDA→CUDA时启用non_blocking，避免跨设备同步报错
    """
    if not isinstance(module, torch.nn.Module):
        logger.warning(f"跳过非nn.Module迁移：{type(module).__name__}")
        return False

    # 修正non_blocking使用条件
    try:
        current_device = next(module.parameters()).device if list(module.parameters()) else torch.device('cpu')
    except StopIteration:
        current_device = torch.device('cpu')
    
    if not (is_cuda_device(current_device) and is_cuda_device(target_device)):
        non_blocking = False

    # 递归迁移所有参数和缓冲区
    try:
        with torch.no_grad():
            # 迁移可训练参数
            for param in module.parameters(recurse=False):
                if param is not None and param.device != target_device:
                    param.data = param.data.to(target_device, non_blocking=non_blocking)
                    if param.grad is not None:
                        param.grad = param.grad.to(target_device, non_blocking=non_blocking)
            # 迁移缓冲区（如running_mean/running_var）
            for buf in module.buffers(recurse=False):
                if buf is not None and buf.device != target_device:
                    buf.data = buf.data.to(target_device, non_blocking=non_blocking)
        return True
    except Exception as e:
        logger.error(f"模块迁移失败：{str(e)}", exc_info=False)
        return False

def normalize_device(device: Any) -> torch.device:
    """统一将设备输入转换为torch.device对象，避免类型混用"""
    if device is None:
        return torch.device('cpu')
    if isinstance(device, int):
        return torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    return torch.device('cpu')

def extract_flux_core(flux_model: Any) -> Optional[torch.nn.Module]:
    """提取Flux模型的核心块结构（single_blocks/double_blocks所在模块）"""
    if not isinstance(flux_model, Flux):
        return None
    
    logger.info(f"🔍 解析Flux模型核心结构...")
    core_candidates: List[str] = ['single_blocks', 'double_blocks', 'model', 'unet', 'inner_model']
    for attr in core_candidates:
        if hasattr(flux_model, attr):
            core = getattr(flux_model, attr)
            logger.info(f"✅ Flux核心属性：.{attr}（类型：{type(core).__name__}）")
            return core
    
    logger.warning(f"⚠️ Flux未找到明确核心属性，使用自身作为候选")
    return flux_model

def find_unet_component(base_model: Any) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    """定位模型中的UNet核心组件（兼容Flux/WAN21/QwenImage）"""
    # Flux模型优先处理
    if isinstance(base_model, Flux) and hasattr(base_model, 'diffusion_model'):
        logger.info("📌 检测到Flux模型，优先解析diffusion_model")
        flux_model = getattr(base_model, 'diffusion_model')
        flux_core = extract_flux_core(flux_model)
        if flux_core and _check_component(flux_core):
            return flux_core, "diffusion_model（Flux核心）"
        elif _check_component(flux_model):
            return flux_model, "diffusion_model（Flux自身）"
    
    # 通用UNet候选匹配
    for attr in UNET_CANDIDATES:
        if hasattr(base_model, attr):
            candidate = getattr(base_model, attr)
            # Flux类型候选额外处理
            if isinstance(candidate, Flux):
                candidate_core = extract_flux_core(candidate)
                if candidate_core and _check_component(candidate_core):
                    return candidate_core, f"{attr}（Flux核心）"
            # 通用块结构检查
            if _check_component(candidate):
                return candidate, attr
    
    return None, None

def _check_component(candidate: Any) -> bool:
    """检查组件是否包含有效块结构（Flux/传统UNet兼容）"""
    if not candidate:
        return False
    
    # Flux特有块检查
    try:
        if hasattr(candidate, 'single_blocks') and isinstance(candidate.single_blocks, (list, torch.nn.ModuleList)):
            logger.info(f"✅ 检测到Flux single_blocks（数量：{len(candidate.single_blocks)}）")
            return True
        if hasattr(candidate, 'double_blocks') and isinstance(candidate.double_blocks, (list, torch.nn.ModuleList)):
            logger.info(f"✅ 检测到Flux double_blocks（数量：{len(candidate.double_blocks)}）")
            return True
    except Exception as e:
        logger.debug(f"Flux块检查异常：{str(e)}")
    
    # 传统UNet块检查
    candidate_attrs = dir(candidate)
    for b_attr in BLOCK_ATTR_CANDIDATES:
        if b_attr in candidate_attrs:
            sub_comp = getattr(candidate, b_attr)
            if isinstance(sub_comp, (list, torch.nn.ModuleList)) or _check_component(sub_comp):
                return True
    
    return False

def find_blocks_attribute(unet: torch.nn.Module) -> Tuple[Optional[str], Optional[List[torch.nn.Module]]]:
    """递归查找UNet中的所有块结构，返回块路径和块列表"""
    block_info: List[Tuple[str, torch.nn.Module]] = []

    def _recursive_find_blocks(obj: Any, parent_path: str = "") -> None:
        if not obj or isinstance(obj, (int, float, str, bool)):
            return
        
        # 优先处理Flux特有块
        for flux_attr in ['single_blocks', 'double_blocks']:
            if hasattr(obj, flux_attr):
                flux_obj = getattr(obj, flux_attr)
                flux_path = f"{parent_path}.{flux_attr}" if parent_path else flux_attr
                if isinstance(flux_obj, (list, torch.nn.ModuleList)):
                    for idx, item in enumerate(flux_obj):
                        item_path = f"{flux_path}[{idx}]"
                        block_info.append((item_path, item))
                _recursive_find_blocks(flux_obj, flux_path)
        
        # 处理传统UNet块
        for b_attr in BLOCK_ATTR_CANDIDATES:
            if hasattr(obj, b_attr) and b_attr not in ['single_blocks', 'double_blocks']:
                b_obj = getattr(obj, b_attr)
                b_path = f"{parent_path}.{b_attr}" if parent_path else b_attr
                if isinstance(b_obj, (list, torch.nn.ModuleList)):
                    for idx, item in enumerate(b_obj):
                        item_path = f"{b_path}[{idx}]"
                        block_info.append((item_path, item))
                _recursive_find_blocks(b_obj, b_path)
    
    _recursive_find_blocks(unet)
    if not block_info:
        logger.warning("❌ 未找到任何有效块结构")
        return None, None
    
    # 排序并提取结果
    block_info.sort(key=lambda x: x[0])
    merged_blocks = [item for (path, item) in block_info]
    full_paths = [path for (path, item) in block_info]
    path_summary = " | ".join(full_paths[:5] + ["..."]) if len(full_paths) > 5 else " | ".join(full_paths)
    
    logger.info(f"📊 找到块总数：{len(merged_blocks)}，块路径示例：{path_summary}")
    return path_summary, merged_blocks

def calculate_vram_metrics() -> Tuple[int, int, float]:
    """计算显存使用指标（已用/总/百分比），兼容多环境"""
    try:
        used = get_system_vram_usage()
        total = get_total_gpu_memory()
    except Exception as e:
        logger.debug(f"外部显存函数调用失败，使用torch备用逻辑：{str(e)}")
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024 ** 2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        else:
            used = -1
            total = -1

    # 数值校验与格式化
    used = max(-1, int(used)) if used is not None else -1
    total = max(1, int(total)) if total is not None and total > 0 else -1
    percent = (used / total * 100) if (total > 0 and used >= 0) else -1.0
    percent = round(max(0.0, min(percent, 100.0)), 1)
    
    return used, total, percent

def get_effective_swap_percent(input_percent: int) -> Tuple[int, bool]:
    """解析交换百分比参数，区分自动/手动模式"""
    if input_percent == -1:
        logger.info(f"🔄 启用自动模式，初始交换百分比={AUTO_INIT_PERCENT}%（显存超标时生效）")
        return AUTO_INIT_PERCENT, True
    
    clamped_percent = max(0, min(input_percent, 100))
    if clamped_percent != input_percent:
        logger.warning(f"📌 交换百分比{input_percent}%超出范围，调整为{clamped_percent}%（手动模式）")
    else:
        logger.info(f"🔧 启用手动模式，固定交换百分比={clamped_percent}%")
    return clamped_percent, False

def calculate_suggested_vr_am_single(pre_gen_used: int, gen_peak_used: int, batch_size: int) -> int:
    """计算单张图片建议额外显存值"""
    if not (torch.cuda.is_available() and pre_gen_used > 0 and gen_peak_used > 0 and batch_size > 0):
        logger.warning("⚠️ 缺少有效计算数据，使用经验默认值")
        return 80
    
    extra_memory = gen_peak_used - pre_gen_used
    if extra_memory <= 0:
        logger.warning(f"⚠️ 显存差值异常（{gen_peak_used} - {pre_gen_used} = {extra_memory}），使用经验默认值")
        return 80
    
    base_vr_am = extra_memory // batch_size
    logger.info(f"📈 基础计算：生成额外显存={extra_memory}MB，批次={batch_size} → 单张基础增量={base_vr_am}MB")
    
    return max(MIN_VR_AM_SINGLE, min(base_vr_am, 1000))

def estimate_memory_usage(vr_am_single: int, batch_size: int) -> int:
    """预估批次图片的显存占用"""
    if vr_am_single <= 0 or batch_size <= 0:
        return 0
    return vr_am_single * batch_size

# ---------------------- 模型结构日志打印节点 ----------------------
class ModelBlockLogger:
    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "log_detailed_structure": ("BOOLEAN", {"default": True, "tooltip": "打印详细结构"}),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "CommonExtension/ModelBlockLogger"
    FUNCTION = "log_model_structure"

    def log_model_structure(self, model: ModelPatcher, log_detailed_structure: bool) -> Tuple[ModelPatcher]:
        def log_blocks(
            model: ModelPatcher, 
            device_to: Any, 
            lowvram_model_memory: Any, 
            force_patch_weights: Any, 
            full_load: Any
        ) -> None:
            """ON_LOAD回调：打印模型结构日志"""
            base_model = model.model
            model_cls_name = type(base_model).__name__
            
            logger.info(f"\n=== 模型结构日志（{model_cls_name}）===")
            logger.info(f"模型类路径: {base_model.__class__.__module__}.{model_cls_name}")
            
            # Flux模型专用信息
            if isinstance(base_model, Flux):
                logger.info("📋 Flux模型专用调试信息：")
                if hasattr(base_model, 'diffusion_model'):
                    flux_model = getattr(base_model, 'diffusion_model')
                    logger.info(f"  diffusion_model类型：{type(flux_model).__name__}")
                    flux_core = extract_flux_core(flux_model)
                    if flux_core:
                        core_attrs = [a for a in dir(flux_core) if not a.startswith('_')]
                        logger.info(f"  Flux核心属性：{core_attrs}")
                        for block_attr in ['single_blocks', 'double_blocks']:
                            if hasattr(flux_core, block_attr):
                                block_obj = getattr(flux_core, block_attr)
                                logger.info(f"    核心.{block_attr}：长度={len(block_obj) if isinstance(block_obj, (list, torch.nn.ModuleList)) else '无'}")
                else:
                    logger.info(f"  未找到diffusion_model属性")
            
            # 基础属性打印
            base_attrs = [a for a in dir(base_model) if not a.startswith('_')]
            logger.info(f"📋 base_model核心属性：{base_attrs}")
            
            # UNet组件识别
            found_unet, found_unet_attr = find_unet_component(base_model)
            if found_unet:
                logger.info(f"✅ 识别到UNet组件：{found_unet_attr}")
                blocks_attr, blocks = find_blocks_attribute(found_unet)
                if blocks_attr and blocks:
                    logger.info(f"📊 块结构：路径={blocks_attr}，总数={len(blocks)}")
                else:
                    logger.warning("⚠️ UNet中未找到有效块结构")
            else:
                logger.error("❌ 未识别到任何UNet组件")
                for b_attr in BLOCK_ATTR_CANDIDATES:
                    if hasattr(base_model, b_attr):
                        obj = getattr(base_model, b_attr)
                        logger.info(f"  base_model.{b_attr}：类型={type(obj).__name__}，长度={len(obj) if isinstance(obj, (list, torch.nn.ModuleList)) else '无'}")
            
            logger.info(f"=== 模型结构日志结束 ===\n")
            comfy.model_management.soft_empty_cache()
            gc.collect()
        
        # 克隆模型并绑定回调
        model_clone = model.clone()
        model_clone.add_callback(CallbacksMP.ON_LOAD, log_blocks)
        return (model_clone, )

# ---------------------- 模型块交换核心节点（优化版） ----------------------
class ModelBlockSwap:
    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (["Auto", "WAN21", "QwenImage", "Flux"], {"default": "Auto", "tooltip": "模型类型"}),
                "blocks_swap_percent": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1, "tooltip": "块交换比例(-1=自动)"}),
                "vram_threshold_percent": ("INT", {"default": 90, "min": 10, "max": 95, "step": 1, "tooltip": "自动模式阈值"}),
                "offload_img_emb": ("BOOLEAN", {"default": False, "tooltip": "卸载图像嵌入层"}),
                "offload_txt_emb": ("BOOLEAN", {"default": False, "tooltip": "卸载文本嵌入层"}),
                "use_non_blocking": ("BOOLEAN", {"default": False, "tooltip": "非阻塞传输"}),
            },
            "optional": {
                "vr_am_single": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 2, "tooltip": "单张额外显存"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1, "tooltip": "批次大小"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "CommonExtension/ModelBlockSwap"
    FUNCTION = "set_callback"

    def set_callback(
        self, 
        model: ModelPatcher, 
        model_type: str, 
        blocks_swap_percent: int, 
        vram_threshold_percent: int, 
        offload_txt_emb: bool, 
        offload_img_emb: bool, 
        use_non_blocking: bool,
        vr_am_single: int = 100,
        batch_size: int = 1
    ) -> Tuple[ModelPatcher]:
        # 初始化核心变量
        current_swap_percent, is_auto_mode = get_effective_swap_percent(blocks_swap_percent)
        estimated_memory = estimate_memory_usage(vr_am_single, batch_size)
        pre_gen_used = -1  # 生成前基础显存
        gen_peak_used = -1  # 生成时峰值显存

        def swap_blocks(
            model: ModelPatcher, 
            device_to: Any, 
            lowvram_model_memory: Any, 
            force_patch_weights: Any, 
            full_load: Any
        ) -> None:
            """ON_LOAD回调：执行模型块交换与嵌入层卸载"""
            nonlocal current_swap_percent, is_auto_mode, estimated_memory
            nonlocal pre_gen_used, gen_peak_used
            
            base_model = model.model
            actual_model_type = model_type if model_type != "Auto" else type(base_model).__name__
            
            # 统一设备格式
            main_device = normalize_device('cuda' if torch.cuda.is_available() else 'cpu')
            offload_device = normalize_device(model.offload_device or 'cpu')
            
            # 前置内存清理
            clean_ram(True, True, False, 1)

            # ---------------------- 自动模式阈值调整 ----------------------
            adjusted_threshold: Optional[float] = None
            if is_auto_mode:
                adjusted_threshold = max(10, min(vram_threshold_percent, 95))
                # 基于预估图片显存调整阈值
                if estimated_memory > 0:
                    _, total_vram, _ = calculate_vram_metrics()
                    if total_vram > 0:
                        estimated_percent = (estimated_memory / total_vram) * 100
                        adjusted_threshold = max(10, min(adjusted_threshold - estimated_percent, 95))
                        logger.info(f"📊 阈值调整：原{vram_threshold_percent}% → 新{adjusted_threshold:.1f}%（预估图片显存{estimated_memory}MB）")
                logger.info(f"📌 自动模式显存阈值：{adjusted_threshold:.1f}%")

            # ---------------------- 模型结构解析 ----------------------
            unet: Optional[torch.nn.Module] = None
            blocks: Optional[List[torch.nn.Module]] = None
            total_blocks = 0
            final_swap_count = 0

            try:
                # Flux手动模式优先解析
                if model_type == "Flux":
                    logger.info(f"\n=== 块交换初始化（Flux手动模式）===")
                    if not hasattr(base_model, 'diffusion_model'):
                        raise ValueError("Flux模型缺少diffusion_model属性")
                    flux_model = getattr(base_model, 'diffusion_model')
                    flux_core = extract_flux_core(flux_model)
                    if not (flux_core and _check_component(flux_core)):
                        raise ValueError("Flux核心无有效块结构")
                    unet = flux_core
                else:
                    logger.info(f"\n=== 块交换初始化（{actual_model_type} - {'自动' if is_auto_mode else '手动'}）===")
                    unet, unet_attr = find_unet_component(base_model)
                    if not unet:
                        raise ValueError("未识别到UNet组件")
                    logger.info(f"✅ UNet组件确认：{unet_attr if 'unet_attr' in locals() else 'Flux核心'}")

                # 提取块列表
                blocks_attr, blocks = find_blocks_attribute(unet)
                if not blocks or len(blocks) == 0:
                    raise ValueError("UNet中无有效块（数量=0）")
                total_blocks = len(blocks)

            except Exception as e:
                logger.error(f"❌ 模型解析失败：{str(e)}", exc_info=False)
                # Flux调试信息补充
                if isinstance(base_model, Flux) and hasattr(base_model, 'diffusion_model'):
                    flux_model = getattr(base_model, 'diffusion_model')
                    logger.info(f"🔍 Flux调试：类名={type(flux_model).__name__}，属性={[a for a in dir(flux_model) if not a.startswith('_')]}")
                return

            # ---------------------- 显存预检（仅自动模式） ----------------------
            # 清理缓存确保预检准确性
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()
            gc.collect()
            pre_check_used, total_vram, pre_check_percent = calculate_vram_metrics()

            # 打印预检信息
            if is_auto_mode and adjusted_threshold is not None:
                logger.info(
                    f"📋 自动模式预检：当前显存={pre_check_used}MB/{total_vram}MB "
                    f"（{pre_check_percent}%），阈值={adjusted_threshold:.1f}%"
                )
            else:
                logger.info(f"📋 显存状态：当前显存={pre_check_used}MB/{total_vram}MB（{pre_check_percent}%）")

            # ---------------------- 块交换执行逻辑 ----------------------
            cycle_count = 0
            # 自动模式且显存达标时跳过交换
            if is_auto_mode and pre_check_percent != -1 and pre_check_percent <= adjusted_threshold:
                logger.info(f"🎉 预检通过：初始显存已低于阈值，无需执行块交换")
                pre_gen_used = pre_check_used
                final_swap_count = 0
            else:
                # 循环执行块交换（直到达标/上限）
                while True:
                    cycle_count += 1
                    swap_count = min(max(0, round(total_blocks * (current_swap_percent / 100))), total_blocks)
                    logger.info(
                        f"🔄 【{actual_model_type} - 第{cycle_count}次交换】"
                        f"百分比={current_swap_percent}% → 交换块数={swap_count}/{total_blocks}"
                    )

                    # 执行块迁移
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    success_count = 0
                    for idx, block in tqdm(enumerate(blocks), total=total_blocks, desc=f"块交换（{current_swap_percent}%）"):
                        target_dev = main_device if idx >= swap_count else offload_device
                        if safe_move_module(block, target_dev, use_non_blocking):
                            success_count += 1

                    # 迁移后清理
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    comfy.model_management.soft_empty_cache()
                    gc.collect()

                    # 更新显存状态
                    pre_gen_used, total_vram, pre_gen_percent = calculate_vram_metrics()
                    logger.info(f"📊 生成前基础显存：{pre_gen_used}MB/{total_vram}MB（{pre_gen_percent}%）")

                    # 退出条件判断
                    if not is_auto_mode:
                        # 手动模式仅执行1次
                        final_swap_count = swap_count
                        break
                    else:
                        # 自动模式：达标/上限则退出
                        if pre_gen_percent != -1 and pre_gen_percent <= adjusted_threshold:
                            logger.info(f"🎉 显存达标（{pre_gen_percent}% ≤ {adjusted_threshold:.1f}%），停止交换")
                            final_swap_count = swap_count
                            break
                        elif current_swap_percent >= 100:
                            logger.warning(f"⚠️ 交换百分比已拉满（100%），显存仍未达标")
                            final_swap_count = swap_count
                            break
                        else:
                            current_swap_percent = min(current_swap_percent + AUTO_SWAP_STEP, 100)

            # ---------------------- 嵌入层卸载 ----------------------
            if offload_txt_emb or offload_img_emb:
                logger.info(f"\n=== {actual_model_type} - 嵌入层卸载 ===")
                # 嵌入层父节点候选
                embed_parents = [unet, base_model]
                if hasattr(base_model, 'conditioner'):
                    embed_parents.insert(0, base_model.conditioner)
                
                # 文本嵌入层卸载
                if offload_txt_emb:
                    self._offload_embedding(embed_parents, DEFAULT_TXT_EMB_ATTR, "文本", offload_device, use_non_blocking)
                # 图像嵌入层卸载
                if offload_img_emb:
                    self._offload_embedding(embed_parents, DEFAULT_IMG_EMB_ATTR, "图像", offload_device, use_non_blocking)

            # ---------------------- 交换结果汇总 ----------------------
            logger.info(f"\n=== {actual_model_type} - 块交换最终结果 ===")
            if is_auto_mode and pre_check_percent != -1 and pre_check_percent <= adjusted_threshold:
                logger.info(f"循环次数：0次（预检通过，未执行交换）")
                model_percent = pre_check_percent
            else:
                logger.info(
                    f"循环次数：{cycle_count}次，最终交换百分比：{current_swap_percent}% "
                    f"交换块数：{final_swap_count}/{total_blocks}"
                )
                model_percent = pre_gen_percent if 'pre_gen_percent' in locals() else pre_check_percent

            # 显存预估汇总
            if total_vram > 0:
                estimated_percent = (estimated_memory / total_vram * 100) if estimated_memory > 0 else 0.0
                total_estimated_percent = model_percent + estimated_percent
                logger.info(f"最终基础显存（模型）：{pre_gen_used}MB/{total_vram}MB（{model_percent}%）")
                if estimated_memory > 0:
                    logger.info(
                        f"预估图片显存：{estimated_memory}MB（{estimated_percent:.1f}%）\n"
                        f"预估总显存（模型+图片）：{total_estimated_percent:.1f}% "
                        f"（{pre_gen_used + estimated_memory}MB/{total_vram}MB）"
                    )
            else:
                logger.warning(f"⚠️ 无法计算显存预估（总显存未知）")
            logger.info(f"====================================\n")

        # 克隆模型并绑定回调
        model_clone = model.clone()
        model_clone.add_callback(CallbacksMP.ON_LOAD, swap_blocks)
        return (model_clone, )
    
    @staticmethod
    def _offload_embedding(
        embed_parents: List[torch.nn.Module],
        emb_attr: str,
        emb_type: str,
        offload_device: torch.device,
        use_non_blocking: bool
    ) -> None:
        """封装嵌入层卸载逻辑，提升代码复用性"""
        emb_obj = None
        found_path = None
        for parent in embed_parents:
            if hasattr(parent, emb_attr):
                emb_obj = getattr(parent, emb_attr)
                found_path = f"{parent.__class__.__name__}.{emb_attr}"
                break
        
        if emb_obj:
            if safe_move_module(emb_obj, offload_device, use_non_blocking):
                logger.info(f"✅ 卸载{emb_type}嵌入层：{found_path}")
            else:
                logger.warning(f"⚠️ {emb_type}嵌入层卸载失败")
        else:
            logger.warning(f"❌ 未找到{emb_type}嵌入层（属性：{emb_attr}）")

# ---------------------- 节点映射 ----------------------
NODE_CLASS_MAPPINGS = {
    "ModelBlockLogger": ModelBlockLogger,
    "ModelBlockSwap": ModelBlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelBlockLogger": "模型Block结构日志打印",
    "ModelBlockSwap": "ModelBlockSwap(-1启用阈值)",
}