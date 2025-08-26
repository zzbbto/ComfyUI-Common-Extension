import torch
import gc
import sys
import os
import time
import ctypes
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Callable
from memory_clear import (
    check_psutil, 
    unload_models, 
    perform_full_cleanup,
    ultimate_cleanup,
    get_system_vram_usage,
    get_total_gpu_memory
)
from types import ModuleType, FunctionType

# ========================== 内存分析器 ==========================

class AnyType(str):
    def __eq__(self, _) -> bool: return True
    def __ne__(self, __value: object) -> bool: return False

any = AnyType("*")

class MemoryAnalyzer:
    def __init__(self):
        self.initial_vram = torch.cuda.memory_allocated()
        self.numCount = 0
        self.history = []
        self.psutil_available = check_psutil()
      
    def _collect_tensors(self, min_size_bytes: float, timeout: float) -> Tuple[List[torch.Tensor], float, int]:
        large_tensors = []
        total_scanned_memory = 0
        small_tensor_count = 0
        start_time = time.time()
        object_count = 0
        max_objects = 500000
        print(f"🔍 开始扫描GPU张量 (最小尺寸: {min_size_bytes/1024**2:.2f}MB)...")
        
        referrers_map = defaultdict(float)
        seen_refs = defaultdict(set)
        
        # 优化：跳过基本类型，提高扫描效率
        skip_types = (int, float, str, bool, type, type(None), ModuleType, FunctionType)
        
        # 新增：跟踪所有张量
        for obj in gc.get_objects():
            if time.time() - start_time > timeout:
                print(f"⏱️ 张量收集超时 ({timeout:.1f}秒)")
                break
                
            object_count += 1
            if object_count % 500000 == 0:
                elapsed = time.time() - start_time
                print(f"  已扫描对象: {object_count}个 | 耗时: {elapsed:.1f}秒")
            
            # 关键修复：封装整个类型检查逻辑
            try:
                # 检查是否可跳过基本类型
                if isinstance(obj, skip_types):
                    continue
                    
                # 原始张量检测逻辑
                tensor = None
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensor = obj
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda:
                    tensor = obj.data
                
                if tensor is not None:
                    tensor_id = id(tensor)
                    tensor_size = tensor.element_size() * tensor.nelement()
                    total_scanned_memory += tensor_size
                    
                    # 区分大小张量
                    if tensor_size >= min_size_bytes:
                        large_tensors.append(tensor)
                        # 引用分析
                        for ref in gc.get_referrers(tensor):
                            ref_id = id(ref)
                            if ref_id not in seen_refs[tensor_id]:
                                seen_refs[tensor_id].add(ref_id)
                                ref_type = str(type(ref))
                                referrers_map[ref_type] += tensor_size
                    else:
                        small_tensor_count += 1
                        
            # 捕获所有异常，防止代理对象初始化失败
            except Exception as e:
                # 可选：记录调试信息
                # print(f"⚠️ 跳过对象 {type(obj)}: {str(e)[:100]}")
                continue
        
        print(f"✅ 找到 {len(large_tensors)} 个大型张量 + {small_tensor_count} 个小型张量")
        if referrers_map:
            print("\n🔗 张量引用分析 (已去重):")
            sorted_refs = sorted(referrers_map.items(), key=lambda x: x[1], reverse=True)[:10]
            for ref_type, total_size in sorted_refs:
                print(f"- {ref_type}: 持有 {total_size/1024**2:.2f} MB 显存")
        
            # 返回大型张量列表 + 总扫描内存 + 小型张量数量
            return large_tensors, total_scanned_memory, small_tensor_count

    def _classify_tensors(self, tensors: List[torch.Tensor]) -> Dict[str, List[Tuple[torch.Tensor, float]]]:
        model_components = defaultdict(list)
        for tensor in tensors:
            try:
                size_bytes = tensor.element_size() * tensor.nelement()
                size_mb = size_bytes / (1024 ** 2)
                comp_type = self._identify_component_type(tensor)
                model_components[comp_type].append((tensor, size_mb))
            except Exception: continue
        return model_components

    def _identify_component_type(self, tensor: torch.Tensor) -> str:
        try:
            shape = tensor.shape
            dtype = str(tensor.dtype)
            
            if len(shape) == 0: return "Scalar"
            if len(shape) == 4 and shape[1] in {3, 4}: return "ImageData"
            if len(shape) == 4 and shape[2] == shape[3] and shape[2] in {8, 16, 32, 64}: return "LatentSpace"
            if len(shape) == 4 and shape[1] in {320, 640, 1280, 2560}: return "UNetFeature"
            if len(shape) == 4 and shape[1] in {16, 32, 64, 96, 128, 256, 512}: return "ConvFeature"
            if len(shape) == 4 and shape[2] == shape[3] and shape[1] % 64 == 0: return "AttentionFeat"
            if len(shape) == 2 and shape[0] == 77 and shape[1] in {768, 1024, 1280}: return "TextEncoding"
            if len(shape) == 1 and shape[0] in {768, 1024, 1280}: return "TextEmbedding"
            if len(shape) == 1 and shape[0] in {1280, 2560}: return "NormParams"
            if len(shape) == 2: return "LinearLayer"
            if len(shape) == 1: return "VectorData"
            if "float" in dtype: return "FloatData"
            if "int" in dtype: return "IntData"
            if "bool" in dtype: return "BoolData"
        except: pass
        return "Unknown"

    def _print_report_header(self, title: str) -> None:
        print("\n" + "="*50)
        print(f"{title}")
        print("="*50)

    def _print_memory_summary(self, current_vram: int, reserved_vram: int, vram_diff_mb: float = 0.0) -> None:
        print(f"初始 VRAM: {self.initial_vram / 1024**2:.2f} MB")
        print(f"当前 VRAM: {current_vram / 1024**2:.2f} MB")
        print(f"保留 VRAM: {reserved_vram / 1024**2:.2f} MB")
        if hasattr(torch.cuda, 'max_memory_allocated'):
            peak_vram = torch.cuda.max_memory_allocated()
            print(f"峰值 VRAM: {peak_vram / 1024**2:.2f} MB")
        if vram_diff_mb != 0.0:
            diff_sign = "+" if vram_diff_mb >= 0 else ""
            print(f"变化量: {diff_sign}{vram_diff_mb:.2f} MB")

# ========================== 集成内存分析器 ==========================

class IntegratedMemoryAnalyzer(MemoryAnalyzer):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "enable_analyzer": ("BOOLEAN", {"default": True}),
                "min_size_mb": ("FLOAT", {"default": 1, "min": 0.1, "max": 1000, "step": 0.1}),
                "timeout": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 300.0, "step": 5.0})
            },
            "optional": {
                "clean_before": ("BOOLEAN", {"default": True}),
                "clean_intensity": ("INT", {"default": 3, "min": 1, "max": 5}),
                "unload": ("BOOLEAN", {"default": True}),
                "print_detail": ("BOOLEAN", {"default": False}),
                "sort_by_size": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "analyze"
    CATEGORY = "CommonExtension/debug"

    def analyze(self, anything: Any,enable_analyzer: bool, min_size_mb: float, timeout: float, 
                clean_before: bool = True, clean_intensity: int = 3, 
                unload: bool = True, print_detail: bool = False, 
                sort_by_size: bool = True) -> Tuple[Any]:
        start_time = time.time()
        self.numCount += 1
        print("\n" + "="*50 + f" 第{self.numCount}次分析开始 " + "="*50)
        
        # 执行清理操作（如果启用）
        if unload:
            unload_models()
        if clean_before:
            print("♻️ 执行清理操作...")
            _, _, pre_vram, post_vram = perform_full_cleanup(
                unload=False,  # 模型已经卸载
                intensity=clean_intensity
            )
            vram_diff_mb = pre_vram - post_vram
            print(f"✅ 清理完成, 释放 {vram_diff_mb:.2f}MB 显存")
        else:
            print("⏩ 跳过清理操作")
            post_vram = get_system_vram_usage()
        
        if not enable_analyzer:
            print("="*50 + f"跳过分析 第{self.numCount}次分析结束 " + "="*50 + "\n")
            return (anything,)

        current_vram = torch.cuda.memory_allocated()
        reserved_vram = torch.cuda.memory_reserved()
        
        min_size_bytes = min_size_mb * 1024 * 1024
        tensors, total_scanned_memory, small_tensor_count = self._collect_tensors(min_size_bytes, timeout)
        model_components = self._classify_tensors(tensors)
        
        self._print_report_header("VRAM 分析报告")
        self._print_memory_summary(current_vram, reserved_vram, 0)
        
        # 内存分布分析
        unscanned_memory = current_vram - total_scanned_memory
        large_tensors_memory = sum(t[1] for tensors in model_components.values() for t in tensors) * (1024 ** 2)
        small_tensors_memory = total_scanned_memory - large_tensors_memory
        
        # 获取系统显存使用情况
        system_vram_usage = get_system_vram_usage()
        total_gpu_memory = get_total_gpu_memory()
        total_usage_percent = (system_vram_usage / total_gpu_memory) * 100
        
        print(f"\n📊 显存分布分析:")
        print(f"- 系统总显存使用: {system_vram_usage:.2f} MB({total_usage_percent:.1f}%)")
        print(f"- PyTorch分配显存: {current_vram/1024**2:.2f} MB({current_vram/1024**2/total_gpu_memory*100:.1f}%)")
        print(f"- 已扫描显存: {total_scanned_memory/1024**2:.2f} MB ({len(tensors)+small_tensor_count}个张量)")
        print(f"- 未追踪显存: {(system_vram_usage * 1024**2 - current_vram)/1024**2:.2f} MB (CUDA上下文/驱动层)")
        print(f"- 大型张量(>={min_size_mb}MB): {large_tensors_memory/1024**2:.2f} MB ({len(tensors)}个)")
        print(f"- 小型张量(<{min_size_mb}MB): {small_tensors_memory/1024**2:.2f} MB ({small_tensor_count}个)")
        
        self._print_component_summary(model_components)
        
        if print_detail:
            self._print_detailed_tensors(model_components, min_size_mb, sort_by_size)
        
        self._print_optimization_suggestions(model_components, system_vram_usage, current_vram/1024**2)
        self._print_history_summary(current_vram)
        self._analyze_fragmentation()
        
        duration = time.time() - start_time
        print(f"✅ 分析完成 (耗时: {duration:.2f}秒)")
        print("="*50 + f" 第{self.numCount}次分析结束 " + "="*50 + "\n")
        
        self.history.append({
            "count": self.numCount,
            "timestamp": time.time(),
            "duration": duration,
            "vram_used": current_vram / (1024 ** 2),
            "system_vram": system_vram_usage,
            "components": {k: len(v) for k, v in model_components.items()}
        })
        
        return (anything,)
    
    def _print_component_summary(self, model_components: Dict[str, List[Tuple[torch.Tensor, float]]]) -> None:
        if not model_components:
            print("\n🔍 未检测到符合条件的张量")
            return
        print("\n📊 显存使用分类:")
        print("-"*50)
        component_totals = []
        for comp_type, tensors in model_components.items():
            total_size = sum(size for _, size in tensors)
            component_totals.append((comp_type, total_size, len(tensors)))
        component_totals.sort(key=lambda x: x[1], reverse=True)
        for comp_type, total_size, count in component_totals:
            print(f"{comp_type:<15}: {count}个张量 | {total_size:>8.2f} MB")
    
    def _print_detailed_tensors(self, model_components: Dict[str, List[Tuple[torch.Tensor, float]]],
                              min_size_mb: float, sort_by_size: bool) -> None:
        print(f"\n🔎 详细张量信息 (大于{min_size_mb}MB):")
        print("-"*90)
        large_tensors = []
        for tensors in model_components.values():
            for tensor, size_mb in tensors:
                if size_mb > min_size_mb:
                    large_tensors.append((tensor, size_mb))
        if sort_by_size:
            large_tensors.sort(key=lambda x: x[1], reverse=True)
        for i, (tensor, size_mb) in enumerate(large_tensors[:50], 1):
            try:
                shape = tuple(tensor.shape)
                dtype = str(tensor.dtype).replace("torch.", "")
                device = str(tensor.device)
                
                # 添加引用信息
                ref_types = defaultdict(int)
                for ref in gc.get_referrers(tensor)[:5]:
                    ref_types[str(type(ref))] += 1
                
                print(f"#{i}: {size_mb:>6.2f} MB | 形状: {str(shape):<20} | 类型: {dtype:<10} | 设备: {device}")
                print(f"    引用: {', '.join([f'{k}×{v}' for k, v in ref_types.items()])}")
            except Exception: pass
    
    def _print_optimization_suggestions(self, 
                                    model_components: Dict[str, List[Tuple[torch.Tensor, float]]],
                                    system_vram: float, pytorch_vram: float):
            print("\n💡 优化建议:")
            current_vram_mb = torch.cuda.memory_allocated() / (1024**2)
            if current_vram_mb > 8000:
                print("- 显存使用接近极限，建议:")
                print("  - 减少批量大小")
                print("  - 使用低精度模式 (FP16)")
                print("  - 优化工作流程")
            if "LatentSpace" in model_components:
                print("- 检测到潜在空间张量，建议使用VAE缩小模式")
            if "ImageData" in model_components:
                print("- 检测到图像数据张量，建议使用图像压缩节点")
            if "UNetFeature" in model_components:
                print("- 检测到UNet特征图，建议使用分块推理")

            if system_vram > 0 and pytorch_vram > 0:
                context_usage = system_vram - pytorch_vram
                context_percent = (context_usage / system_vram) * 100
                
                print(f"\n🔧 CUDA上下文分析:")
                print(f"- CUDA上下文占用: {context_usage:.2f} MB ({context_percent:.1f}%)")
                
                # if context_percent > 30:
                #     print("⚠️ CUDA上下文占用过高，建议:")
                #     print("  - 减少工作流中模型加载/卸载次数")
                #     print("  - 使用'内存清理'节点")
                #     print("  - 检查是否有其他程序占用显存")
                # elif context_percent > 15:
                #     print("💡 CUDA上下文占用适中，可考虑优化工作流")            
    
    def _print_history_summary(self, current_vram: int) -> None:
        if len(self.history) > 1 and len(self.history) >= 2:
            last_vram = self.history[-2]["vram_used"]
            current_vram_mb = current_vram / (1024**2)
            trend = "增加" if current_vram_mb > last_vram else "减少"
            diff = abs(current_vram_mb - last_vram)
            print(f"\n📈 历史趋势: 相比上次分析显存{trend} {diff:.2f}MB")

            
    def _analyze_fragmentation(self) -> None:
        try:
            if hasattr(torch.cuda, 'memory_stats'):
                device = torch.cuda.current_device()
                stats = torch.cuda.memory_stats(device=device)
                if 'bytes' in stats and 'active_bytes.all.current' in stats:
                    total_bytes = stats['bytes']
                    active_bytes = stats['active_bytes.all.current']
                    free_bytes = total_bytes - active_bytes
                    max_free_block = stats.get('segment.max_free_bytes', free_bytes)
                    if free_bytes > 0:
                        fragmentation = 1.0 - (max_free_block / free_bytes)
                        print(f"\n🧩 显存碎片率: {fragmentation*100:.2f}%")
                        print(f"- 总空闲显存: {free_bytes/1024**2:.2f} MB")
                        print(f"- 最大空闲块: {max_free_block/1024**2:.2f} MB")
                        if fragmentation > 0.5:
                            print("⚠️ 高碎片率! 建议:")
                            print("  - 使用高强度清理")
                            print("  - 重启ComfyUI")
                        elif fragmentation > 0.3:
                            print("💡 中等碎片率，可能影响大模型加载")
                    else:
                        print("\n🧩 无空闲显存可用于碎片分析")
        except Exception as e:
            print(f"⚠️ 碎片分析失败: {str(e)}")

# ========================== 内存清理器 ==========================

class MemoryCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {              
                "anything": (any, {}),
                "clean_intensity": ("INT", {"default": 3, "min": 1, "max": 5}),
                "unload": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "clean"
    CATEGORY = "CommonExtension/debug"

    def clean(self, anything: Any, clean_intensity: int, unload: bool) -> Tuple[Any]:
        
        if unload:
            unload_models()

        if clean_intensity == 5:
            # 终极清理有特殊处理
            print("🚀 执行终极内存清理...")
            pre_ram = get_system_vram_usage()
            result = ultimate_cleanup(warn=False)
            post_ram = get_system_vram_usage()
            
            self._print_clean_report(pre_ram, post_ram, result, clean_intensity)
        else:
            # 常规清理
            print(f"🧹 执行{clean_intensity}级内存清理...")
            pre_ram, post_ram, pre_vram, post_vram = perform_full_cleanup(
                unload=False,
                intensity=clean_intensity
            )
            freed = pre_vram - post_vram
            self._print_clean_report(pre_vram, post_vram, freed, clean_intensity)
        
        return (anything,)
    
    def _print_clean_report(self, pre: float, post: float, result: float, clean_intensity: int) -> None:
        print("\n" + "="*50 + " 内存清理报告 " + "="*50)
        
        if clean_intensity == 5:
            if result is not None:
                freed = pre - post
                print(f"✅ 终极清理成功完成!")
                print(f"释放显存: {freed:.2f} MB (前: {pre:.2f} MB → 后: {post:.2f} MB)")
            else:
                print("⚠️ 终极清理失败，但已尝试恢复基本功能")
                print("建议重启ComfyUI以完全恢复")
        else:
            freed = pre - post
            print(f"✅ {clean_intensity}级清理完成!")
            print(f"释放显存: {freed:.2f} MB (前: {pre:.2f} MB → 后: {post:.2f} MB)")
        
        print("="*90 + "\n")

# ========================== 节点注册 ==========================

NODE_CLASS_MAPPINGS = {
    "IntegratedMemoryAnalyzer": IntegratedMemoryAnalyzer,
    "MemoryCleaner": MemoryCleaner 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntegratedMemoryAnalyzer": "集成内存分析器",
    "MemoryCleaner": "内存清理器"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']