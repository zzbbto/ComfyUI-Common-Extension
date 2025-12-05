import comfy.model_management
import gc
import torch
import logging
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher
from comfy.model_base import WAN21, QwenImage, Flux
from tqdm import tqdm
from memory_clear import (
    get_system_vram_usage,
    get_total_gpu_memory
)
from memory_util import (
    clean_ram,
)

# ---------------------- 共享常量与工具函数 ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [BlockSwap] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ModelBlockLogger")

UNET_CANDIDATES = ['unet', 'diffusion_model', 'model', 'denoise_fn', 'network', 'transformer_unet']
BLOCK_ATTR_CANDIDATES = ['blocks', 'transformer_blocks', 'down', 'up', 'mid', 'layers', 'resblocks', 
                         'single_blocks', 'double_blocks']
AUTO_INIT_PERCENT = 20
AUTO_SWAP_STEP = 20
DEFAULT_TXT_EMB_ATTR = "text_embedding"
DEFAULT_IMG_EMB_ATTR = "img_emb"
MIN_VR_AM_SINGLE = 40

# 新增：检查设备是否为CUDA（统一判断逻辑）
def is_cuda_device(device):
    if isinstance(device, str):
        return device.startswith('cuda')
    elif isinstance(device, torch.device):
        return device.type == 'cuda'
    return False

# 新增：安全迁移模块到目标设备（核心修复CUDA报错）
def safe_move_module(module, target_device, non_blocking=False):
    """
    安全迁移模块的参数/缓冲区到目标设备，避免直接module.to()导致的CUDA invalid argument错误
    :param module: nn.Module 待迁移模块
    :param target_device: torch.device 目标设备
    :param non_blocking: bool 非阻塞传输（仅CUDA→CUDA生效）
    :return: bool 是否迁移成功
    """
    # 1. 校验模块类型
    if not isinstance(module, torch.nn.Module):
        logger.warning(f"跳过非nn.Module迁移：{type(module).__name__}")
        return False
    
    # 2. 修正non_blocking：仅CUDA→CUDA时启用
    current_device = next(module.parameters()).device if len(list(module.parameters())) > 0 else torch.device('cpu')
    if not (is_cuda_device(current_device) and is_cuda_device(target_device)):
        non_blocking = False
    
    # 3. 逐参数迁移（避免整模块to()的兼容性问题）
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
        logger.error(f"模块迁移失败：{str(e)}")
        return False

# 新增：统一设备为torch.device对象（避免类型混用）
def normalize_device(device):
    if device is None:
        return torch.device('cpu')
    if isinstance(device, int):
        return torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    return torch.device('cpu')

def extract_flux_core(flux_model):
    if not hasattr(flux_model, '__class__') or flux_model.__class__.__name__ != "Flux":
        return None
    
    logger.info(f"🔍 解析Flux模型，尝试提取核心结构...")
    core_candidates = ['single_blocks', 'double_blocks', 'model', 'unet', 'inner_model']
    for attr in core_candidates:
        if hasattr(flux_model, attr):
            core = getattr(flux_model, attr)
            logger.info(f"✅ Flux找到核心属性：.{attr}（类型：{type(core).__name__}）")
            core_attrs = [a for a in dir(core) if not a.startswith('_')]
            logger.debug(f"Flux核心属性.{attr}的子属性：{core_attrs}")
            return core
    
    logger.warning(f"⚠️ Flux未找到明确核心属性，使用自身作为候选")
    flux_attrs = [a for a in dir(flux_model) if not a.startswith('_')]
    logger.debug(f"Flux自身属性：{flux_attrs}")
    return flux_model

def find_unet_component(base_model):
    if isinstance(base_model, Flux) and hasattr(base_model, 'diffusion_model'):
        logger.info("📌 检测到Flux模型，优先处理diffusion_model（类型为Flux）")
        flux_model = getattr(base_model, 'diffusion_model')
        flux_core = extract_flux_core(flux_model)
        if flux_core:
            if _check_component(flux_core):
                return flux_core, "diffusion_model（Flux核心）"
            elif _check_component(flux_model):
                return flux_model, "diffusion_model（Flux自身）"
    
    for attr in UNET_CANDIDATES:
        if hasattr(base_model, attr):
            candidate = getattr(base_model, attr)
            if hasattr(candidate, '__class__') and candidate.__class__.__name__ == "Flux":
                candidate_core = extract_flux_core(candidate)
                if candidate_core and _check_component(candidate_core):
                    return candidate_core, f"{attr}（Flux核心）"
            if _check_component(candidate):
                return candidate, attr
    
    return None, None

def _check_component(candidate):
    if not candidate:
        return False
    
    candidate_attrs = dir(candidate)
    has_flux_blocks = any(attr in candidate_attrs for attr in ['single_blocks', 'double_blocks'])
    if has_flux_blocks:
        try:
            if hasattr(candidate, 'single_blocks') and hasattr(candidate.single_blocks, '__len__') and hasattr(candidate.single_blocks, '__iter__'):
                logger.info(f"✅ 检测到Flux风格块结构（single_blocks，数量：{len(candidate.single_blocks)}）")
                return True
            if hasattr(candidate, 'double_blocks') and hasattr(candidate.double_blocks, '__len__') and hasattr(candidate.double_blocks, '__iter__'):
                logger.info(f"✅ 检测到Flux风格块结构（double_blocks，数量：{len(candidate.double_blocks)}）")
                return True
        except Exception as e:
            logger.debug(f"检查Flux块时出错：{str(e)}")
    
    for b_attr in BLOCK_ATTR_CANDIDATES:
        if b_attr in candidate_attrs:
            sub_comp = getattr(candidate, b_attr)
            if (hasattr(sub_comp, '__len__') and hasattr(sub_comp, '__iter__')) or _check_component(sub_comp):
                return True
    
    return False

def find_blocks_attribute(unet):
    block_info = []
    
    def _recursive_find_blocks(obj, parent_path=""):
        if not obj or isinstance(obj, (int, float, str, bool)):
            return
        
        for flux_attr in ['single_blocks', 'double_blocks']:
            if hasattr(obj, flux_attr):
                flux_obj = getattr(obj, flux_attr)
                flux_path = f"{parent_path}.{flux_attr}" if parent_path else flux_attr
                logger.debug(f"🔍 扫描Flux特有块：{flux_path}（类型：{type(flux_obj).__name__}）")
                if hasattr(flux_obj, '__len__') and hasattr(flux_obj, '__iter__'):
                    for idx, item in enumerate(flux_obj):
                        item_path = f"{flux_path}[{idx}]"
                        block_info.append((item_path, item))
                        logger.debug(f"✅ 找到Flux块：{item_path}（类型：{type(item).__name__}）")
                _recursive_find_blocks(flux_obj, flux_path)
        
        for b_attr in BLOCK_ATTR_CANDIDATES:
            if hasattr(obj, b_attr) and b_attr not in ['single_blocks', 'double_blocks']:
                b_obj = getattr(obj, b_attr)
                b_path = f"{parent_path}.{b_attr}" if parent_path else b_attr
                if hasattr(b_obj, '__len__') and hasattr(b_obj, '__iter__'):
                    for idx, item in enumerate(b_obj):
                        item_path = f"{b_path}[{idx}]"
                        block_info.append((item_path, item))
                        logger.debug(f"✅ 找到传统块：{item_path}（类型：{type(item).__name__}）")
                _recursive_find_blocks(b_obj, b_path)
    
    _recursive_find_blocks(unet)
    if not block_info:
        logger.warning("❌ 未找到任何有效块结构")
        return None, None
    
    block_info.sort(key=lambda x: x[0])
    merged_blocks = [item for (path, item) in block_info]
    full_paths = [path for (path, item) in block_info]
    
    logger.info(f"📊 找到块总数：{len(merged_blocks)}，块路径示例：{full_paths[:3]}...")
    return " | ".join(full_paths[:5] + ["..."] if len(full_paths) > 5 else full_paths), merged_blocks

def calculate_vram_metrics():
    try:
        used = get_system_vram_usage()
        total = get_total_gpu_memory()
    except Exception as e:
        logger.debug(f"外部显存函数失败，用torch备用：{str(e)}")
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024 ** 2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        else:
            used = -1
            total = -1

    used = max(-1, int(used)) if used is not None else -1
    total = max(1, int(total)) if total is not None and total > 0 else -1
    percent = (used / total * 100) if (total > 0 and used >= 0) else -1.0
    
    if used == -1:
        logger.warning("⚠️  无法获取当前显存占用")
    if total == -1:
        logger.warning("⚠️  无法获取总显存")
    
    return used, total, round(max(0.0, min(percent, 100.0)), 1)

def get_effective_swap_percent(input_percent):
    if input_percent == -1:
        logger.info(f"🔄 启用自动模式，初始交换百分比={AUTO_INIT_PERCENT}%（显存超标时生效）")
        return AUTO_INIT_PERCENT, True
    clamped_percent = max(0, min(input_percent, 100))
    if clamped_percent != input_percent:
        logger.warning(f"📌 交换百分比{input_percent}%超出范围，调整为{clamped_percent}%（手动模式）")
    logger.info(f"🔧 启用手动模式，固定交换百分比={clamped_percent}%")
    return clamped_percent, False

def calculate_suggested_vr_am_single(pre_gen_used, gen_peak_used, batch_size):
    if not (torch.cuda.is_available() and pre_gen_used > 0 and gen_peak_used > 0 and batch_size > 0):
        logger.warning("⚠️ 缺少有效计算数据（CUDA未启用/显存数据异常/批次为0），使用经验默认值")
        return 80
    
    extra_memory = gen_peak_used - pre_gen_used
    if extra_memory <= 0:
        logger.warning(f"⚠️ 显存差值异常（{gen_peak_used} - {pre_gen_used} = {extra_memory}），使用经验默认值")
        return 80
    
    base_vr_am = extra_memory // batch_size
    logger.info(f"📈 基础计算：生成额外显存={extra_memory}MB，批次={batch_size} → 单张基础增量={base_vr_am}MB")
    
    suggested = max(MIN_VR_AM_SINGLE, min(base_vr_am, 1000))
    return suggested

def estimate_memory_usage(vr_am_single, batch_size):
    if vr_am_single <= 0 or batch_size <= 0:
        return 0
    estimated_memory = vr_am_single * batch_size
    return estimated_memory

# ---------------------- ModelBlockLogger 类 ----------------------
class ModelBlockLogger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "log_detailed_structure": ("BOOLEAN", {"default": True, "tooltip": "打印详细结构"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "CommonExtension/ModelBlockLogger"
    FUNCTION = "log_model_structure"

    def log_model_structure(self, model: ModelPatcher, log_detailed_structure):
        def log_blocks(model: ModelPatcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            base_model = model.model
            model_cls_name = type(base_model).__name__
            
            logger.info(f"\n=== 模型结构日志（{model_cls_name}）===")
            logger.info(f"模型类路径: {base_model.__class__.__module__}.{model_cls_name}")
            
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
                                logger.info(f"    核心.{block_attr}：长度={len(block_obj) if hasattr(block_obj, '__len__') else '无'}")
                else:
                    logger.info(f"  未找到diffusion_model属性")
            
            base_attrs = [a for a in dir(base_model) if not a.startswith('_')]
            logger.info(f"📋 base_model核心属性：{base_attrs}")
            
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
                        logger.info(f"  base_model.{b_attr}：类型={type(obj).__name__}，长度={len(obj) if hasattr(obj, '__len__') else '无'}")
            
            logger.info(f"=== 模型结构日志结束 ===\n")
            comfy.model_management.soft_empty_cache()
            gc.collect()
        
        model = model.clone()
        model.add_callback(CallbacksMP.ON_LOAD, log_blocks)
        return (model, )

# ---------------------- ModelBlockSwap 类（核心修改） ----------------------
class ModelBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (["Auto", "WAN21", "QwenImage", "Flux"], {"default": "Auto", "tooltip": "模型类型"}),
                "blocks_swap_percent": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1, "tooltip": "块交换比例(-1=自动)，控制移至CPU的块占比"}),
                "vram_threshold_percent": ("INT", {"default": 90, "min": 10, "max": 95, "step": 1, "tooltip": "自动模式阈值"}),
                "offload_img_emb": ("BOOLEAN", {"default": False, "tooltip": "卸载图像嵌入层"}),
                "offload_txt_emb": ("BOOLEAN", {"default": False, "tooltip": "卸载文本嵌入层"}),
                "use_non_blocking": ("BOOLEAN", {"default": False, "tooltip": "非阻塞传输"}),
            },
            "optional": {
                "vr_am_single": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 2, "tooltip": "单张额外显存"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1, "tooltip": "批次大小（≥1）"}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "CommonExtension/ModelBlockSwap"
    FUNCTION = "set_callback"

    def set_callback(
        self, model: ModelPatcher, model_type, blocks_swap_percent, 
        vram_threshold_percent, offload_txt_emb, offload_img_emb, use_non_blocking,
        vr_am_single=100, batch_size=1
    ):
        current_swap_pct, is_auto_mode = get_effective_swap_percent(blocks_swap_percent)
        estimated_memory = estimate_memory_usage(vr_am_single, batch_size)

        # 新增：存储生成前后的显存数据（跨回调使用）
        pre_gen_used = -1  # 生成前基础显存
        gen_peak_used = -1  # 生成时峰值显存

        def swap_blocks(model: ModelPatcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            nonlocal current_swap_pct, is_auto_mode, estimated_memory
            nonlocal pre_gen_used, gen_peak_used  # 声明为非局部变量
            base_model = model.model
            actual_model_type = model_type if model_type != "Auto" else type(base_model).__name__
            
            # 修复：统一设备为torch.device对象
            main_device = normalize_device('cuda' if torch.cuda.is_available() else 'cpu')
            offload_device = normalize_device(model.offload_device or 'cpu')
            
            # 清理RAM（避免内存泄漏）
            clean_ram(True, True, False, 1)
            
            # ---------------------- 核心修改1：仅自动模式处理阈值 ----------------------
            adjusted_threshold = None
            if is_auto_mode:
                # 自动模式：基于原始参数初始化阈值（避免None）
                adjusted_threshold = max(10, min(vram_threshold_percent, 95))
                # 仅自动模式下计算预估显存调整
                if estimated_memory > 0:
                    _, total_vram, _ = calculate_vram_metrics()
                    if total_vram > 0:
                        estimated_percent = (estimated_memory / total_vram) * 100
                        # 用调整后的阈值做减法（此时adjusted_threshold必为数值）
                        adjusted_threshold = max(10, min(adjusted_threshold - estimated_percent, 95))
                        logger.info(f"📊 阈值调整：原{vram_threshold_percent}% → 新{adjusted_threshold:.1f}%（预估图片显存{estimated_memory:.2f}MB）")
                logger.info(f"📌 自动模式显存阈值：{adjusted_threshold:.1f}%")

            # 2. 模型结构解析（Flux优先）
            unet = None
            blocks = None
            total_blocks = 0
            final_swap_count = 0  # 初始化：修复未定义问题
            try:
                if model_type == "Flux":
                    logger.info(f"\n=== 块交换初始化（Flux手动模式）===")
                    if not hasattr(base_model, 'diffusion_model'):
                        raise ValueError("Flux模型缺少diffusion_model属性")
                    flux_model = getattr(base_model, 'diffusion_model')
                    flux_core = extract_flux_core(flux_model)
                    if not (flux_core and _check_component(flux_core)):
                        raise ValueError("Flux核心无有效块结构")
                    unet = flux_core
                    logger.info(f"✅ Flux结构解析完成：diffusion_model → 核心块")
                else:
                    logger.info(f"\n=== 块交换初始化（{actual_model_type} - {'自动' if is_auto_mode else '手动'}）===")
                    unet, unet_attr = find_unet_component(base_model)
                    if not unet:
                        raise ValueError("未识别到UNet组件")

                # 提取块结构
                blocks_attr, blocks = find_blocks_attribute(unet)
                if not blocks or len(blocks) == 0:
                    raise ValueError("UNet中无有效块（数量=0）")
                total_blocks = len(blocks)
                logger.info(f"📊 结构确认：UNet来源={unet_attr if 'unet_attr' in locals() else 'Flux核心'}，块总数={total_blocks}")

            except Exception as e:
                logger.error(f"❌ 模型解析失败：{str(e)}")
                if isinstance(base_model, Flux) and hasattr(base_model, 'diffusion_model'):
                    flux_model = getattr(base_model, 'diffusion_model')
                    logger.info(f"🔍 Flux调试：类名={flux_model.__class__.__name__}，属性={[a for a in dir(flux_model) if not a.startswith('_')]}")
                return

            # ---------------------- 核心修改2：自动模式新增【交换前显存预检】 ----------------------
            # 先清理缓存并获取当前显存（排除历史残留）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 新增：确保缓存清理完成
            comfy.model_management.soft_empty_cache()
            gc.collect()
            pre_check_used, total_vram, pre_check_percent = calculate_vram_metrics()
            
            # 修正此处的日志输出，增加条件判断
            if is_auto_mode and adjusted_threshold is not None:
                logger.info(f"📋 自动模式预检：当前显存占用={pre_check_used}MB/总{total_vram}MB（{pre_check_percent}%），阈值={adjusted_threshold:.1f}%")
            else:
                logger.info(f"📋 显存状态：当前显存占用={pre_check_used}MB/总{total_vram}MB（{pre_check_percent}%）")

            cycle_count = 0  
            target_met = False
            # 预检逻辑：仅自动模式且显存已达标时，跳过交换
            if is_auto_mode and pre_check_percent != -1 and pre_check_percent <= adjusted_threshold:
                logger.info(f"🎉 预检通过：初始显存已低于阈值，无需执行块交换")
                pre_gen_used = pre_check_used  # 复用预检的显存数据
                final_swap_count = 0  # 无交换，块数为0
            else:
                # 3. 块交换执行（仅预检不通过时进入）
                while True:
                    cycle_count += 1
                    swap_count = min(max(0, round(total_blocks * (current_swap_pct / 100))), total_blocks)
                    logger.info(f"🔄 【{actual_model_type} - 第{cycle_count}次交换】百分比={current_swap_pct}% → 交换块数={swap_count}/{total_blocks}")

                    # 块迁移（核心修复：替换block.to()为safe_move_module）
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # 新增：避免异步操作冲突
                    success_count = 0
                    for idx, block in tqdm(enumerate(blocks), total=total_blocks, desc=f"块交换（{current_swap_pct}%）"):
                        target_dev = main_device if idx >= swap_count else offload_device
                        try:
                            # 修复：使用安全迁移函数替代直接to()
                            if safe_move_module(block, target_dev, use_non_blocking):
                                success_count += 1
                            else:
                                logger.warning(f"⚠️ 块{idx}迁移失败（安全迁移函数返回False），跳过")
                        except Exception as e:
                            logger.warning(f"⚠️ 块{idx}迁移失败：{str(e)}，跳过")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # 新增：确保所有迁移完成

                    # 清理缓存 + 记录生成前基础显存
                    comfy.model_management.soft_empty_cache()
                    gc.collect()
                    pre_gen_used, total_vram, pre_gen_percent = calculate_vram_metrics()
                    logger.info(f"📊 【生成前基础显存】已用{pre_gen_used}MB/总{total_vram}MB（{pre_gen_percent}%）")

                    # ---------------------- 核心修改3：仅自动模式判断阈值终止 ----------------------
                    if not is_auto_mode:
                        # 非自动模式：只执行1次交换，直接退出循环
                        final_swap_count = swap_count
                        break
                    else:
                        # 自动模式：基于调整后的阈值判断是否终止
                        if pre_gen_percent != -1 and pre_gen_percent <= adjusted_threshold:
                            logger.info(f"🎉 显存达标（{pre_gen_percent}% ≤ {adjusted_threshold:.1f}%），停止交换")
                            target_met = True
                            final_swap_count = swap_count
                            break
                        elif current_swap_pct >= 100:
                            logger.warning(f"⚠️ 交换百分比已拉满（100%），显存仍未达标")
                            final_swap_count = swap_count
                            break
                        else:
                            current_swap_pct = min(current_swap_pct + AUTO_SWAP_STEP, 100)

            # 5. 嵌入层卸载（修复：使用安全迁移函数）
            if offload_txt_emb or offload_img_emb:
                logger.info(f"\n=== {actual_model_type} - 嵌入层卸载 ===")
                embed_parents = [unet, base_model]
                if hasattr(base_model, 'conditioner'):
                    embed_parents.insert(0, base_model.conditioner)
                
                for emb_type, emb_attr in [("文本", DEFAULT_TXT_EMB_ATTR), ("图像", DEFAULT_IMG_EMB_ATTR)]:
                    if (emb_type == "文本" and not offload_txt_emb) or (emb_type == "图像" and not offload_img_emb):
                        continue
                    emb_obj = None
                    found_path = None
                    for parent in embed_parents:
                        if hasattr(parent, emb_attr):
                            emb_obj = getattr(parent, emb_attr)
                            found_path = f"{parent.__class__.__name__}.{emb_attr}"
                            break
                    if emb_obj:
                        try:
                            # 修复：使用安全迁移函数卸载嵌入层
                            if safe_move_module(emb_obj, offload_device, use_non_blocking):
                                logger.info(f"✅ 卸载{emb_type}嵌入层：{found_path}")
                            else:
                                logger.warning(f"⚠️ {emb_type}嵌入层卸载失败（安全迁移函数返回False）")
                        except Exception as e:
                            logger.warning(f"⚠️ {emb_type}嵌入层卸载失败：{str(e)}")
                    else:
                        logger.warning(f"❌ 未找到{emb_type}嵌入层（属性：{emb_attr}）")

            # 6. 交换结果日志
            logger.info(f"\n=== {actual_model_type} - 块交换最终结果 ===")
            if is_auto_mode and pre_check_percent != -1 and pre_check_percent <= adjusted_threshold:
                logger.info(f"循环次数：0次（预检通过，未执行交换）")
                # 预检通过：用“预检时的模型百分比 + 图片预估百分比”
                model_percent = pre_check_percent
            else:
                logger.info(f"循环次数：{cycle_count}次，最终交换百分比：{current_swap_pct}%，交换块数：{final_swap_count}/{total_blocks}")
                # 预检不通过：用“交换后的模型百分比 + 图片预估百分比”
                model_percent = pre_gen_percent if 'pre_gen_percent' in locals() else pre_check_percent

            # 统一计算图片预估百分比（避免重复代码）
            estimated_percent_calc = (estimated_memory / total_vram * 100) if (total_vram > 0 and estimated_memory > 0) else 0.0
            # 计算总预估百分比（模型 + 图片）
            total_estimated_percent = model_percent + estimated_percent_calc

            # 打印最终结果（区分是否能计算总显存）
            logger.info(f"最终基础显存（模型）：{pre_gen_used}MB/总{total_vram}MB（{model_percent}%）")
            if estimated_memory > 0 and total_vram > 0:
                logger.info(f"预估图片显存：{estimated_memory:.2f}MB（{estimated_percent_calc:.1f}%）")
                logger.info(f"预估总显存（模型+图片）：{total_estimated_percent:.1f}%（{pre_gen_used + estimated_memory:.2f}MB/总{total_vram}MB）")
            else:
                logger.warning(f"⚠️ 无法计算预估总显存（总显存未知或图片预估显存为0）")
            logger.info(f"====================================\n")

        # 克隆模型并绑定块交换回调
        model = model.clone()
        model.add_callback(CallbacksMP.ON_LOAD, swap_blocks)
        return (model, )

# ---------------------- 节点映射 ----------------------
NODE_CLASS_MAPPINGS = {
    "ModelBlockLogger": ModelBlockLogger,
    "ModelBlockSwap": ModelBlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelBlockLogger": "模型Block结构日志打印",
    "ModelBlockSwap": "ModelBlockSwap(-1启用阈值)",
}