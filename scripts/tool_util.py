import math
import torch
import gc
import sys
import os
import time
import numpy as np
from collections import defaultdict
from comfy import model_management
from typing import Any, Dict, List, Tuple, Optional

class AnyType(str):
    """用于表示任意类型的特殊类，在类型比较时总是返回相等"""
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class AnyBatchSliceNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "batch_data": (any, {}),      # 支持任意可切片类型
                "start_index": ("INT", {       # 起始索引
                    "default": 0, 
                    "min": 0,
                    "max": 999999,
                    "step": 1
                }),
                "length": ("INT", {            # 切片长度
                    "default": 1,
                    "min": -1,                # -1 表示取到末尾
                    "max": 999999,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = (any,)    
    RETURN_NAMES = ("sliced_data",)
    FUNCTION = "do_slice"
    CATEGORY = "CommonExtension/tool"
    DESCRIPTION = "对任意可切片类型的数据进行批处理切片操作"

    def do_slice(self, batch_data: Any, start_index: int, length: int) -> Tuple[Any]:
        # 类型检查
        if not hasattr(batch_data, '__getitem__') or not hasattr(batch_data, '__len__'):
            raise ValueError("❌ 输入数据不支持切片操作")
        
        data_len = len(batch_data)
        
        # 处理负值逻辑
        if start_index >= data_len:
            return (batch_data.__class__(),)  # 返回空切片
        
        if length == -1:
            # 取到末尾
            length = data_len - start_index
        elif length < 1:
            raise ValueError(f"❌ 无效的长度值: {length} (必须为-1或正整数)")
        
        # 计算结束索引
        end_index = min(start_index + length, data_len)
        
        # 执行切片操作
        try:
            sliced = batch_data[start_index:end_index]
        except Exception as e:
            raise RuntimeError(f"❌ 切片操作失败: {str(e)}")
        
        # 特殊处理 Torch Tensor 的梯度
        if isinstance(sliced, torch.Tensor):
            sliced = sliced.detach().clone()
        
        print(f"✅ 切片成功: 索引 {start_index}-{end_index} (长度: {len(sliced)}/{data_len})")
        return (sliced,)


    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "anything": (any, {}),
                "aggressive": ("BOOLEAN", {"default": True}),
                "unload_models": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "clean"
    CATEGORY = "CommonExtension/debug"
    DESCRIPTION = "强制清理内存和显存，解决内存泄漏问题"

    def __init__(self):
        self.psutil_available = self.check_psutil()
    
    def check_psutil(self) -> bool:
        try:
            import psutil
            return True
        except ImportError:
            print("⚠️ psutil 未安装，无法获取系统内存信息")
            return False

    def clean(self, anything: Any, aggressive: bool, unload_models: bool) -> Tuple[Any]:
        # 记录清理前的内存状态
        pre_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        pre_ram = self.get_system_memory_usage() if self.psutil_available else 0.0
        
        # 标准清理
        if unload_models:
            model_management.unload_all_models()
        
        # 清理PyTorch缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 激进模式清理
        if aggressive:
            # 强制Python垃圾回收
            for i in range(3):  # 多次回收确保清理
                gc.collect()
            
            # 释放所有缓存的分配器内存
            if hasattr(torch.cuda, 'memory_reserved'):
                torch.cuda.memory_reserved(0)
            
            # 特殊处理ComfyUI内部缓存
            self.clear_comfy_caches()
        
        # 记录清理后的内存状态
        post_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        post_ram = self.get_system_memory_usage() if self.psutil_available else 0.0
        
        # 打印清理结果
        print("\n" + "="*50 + " 内存清理报告 " + "="*50)
        print(f"显存释放: {pre_vram - post_vram:.2f} MB (前: {pre_vram:.2f} MB → 后: {post_vram:.2f} MB)")
        if self.psutil_available:
            print(f"系统内存释放: {pre_ram - post_ram:.2f} MB (前: {pre_ram:.2f} MB → 后: {post_ram:.2f} MB)")
        print("="*90 + "\n")
        
        return (anything,)

    def get_system_memory_usage(self) -> float:
        """获取当前进程的系统内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)
        except Exception:
            return 0.0

    def clear_comfy_caches(self) -> None:
        """清除ComfyUI内部缓存"""
        # 尝试清除已知缓存
        cache_cleared = 0
        for module_name, module in list(sys.modules.items()):
            if 'comfy' in module_name or 'nodes' in module_name:
                try:
                    if hasattr(module, 'cache_clear'):
                        module.cache_clear()
                        cache_cleared += 1
                except Exception:
                    pass
        
        # 清除模型加载器缓存
        if 'model_management' in sys.modules:
            try:
                model_management_module = sys.modules['model_management']
                if hasattr(model_management_module, 'loaded_models'):
                    model_management_module.loaded_models = []
                    cache_cleared += 1
            except Exception:
                pass
        
        print(f"♻️ 清理了 {cache_cleared} 个内部缓存")

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AnyBatchSliceNode": AnyBatchSliceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyBatchSliceNode": "通用批处理切片",
}

# 导出配置
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']