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
    
class SegmentFrameAdjuster:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_frames": ("STRING", {"default": ""}),
                "max_frames": ("INT", {"default": 81, "min": 1, "max": 1000}),
                "min_frames": ("INT", {"default": 25, "min": 1, "max": 1000}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("adjusted_segments", "total_frames", "segment_count", "segment_frames_index", "segment_frames_index_next")
    FUNCTION = "adjust_segments"
    CATEGORY = "CommonExtension/tool"

    def adjust_segments(self, segment_frames, max_frames, min_frames):
        # 1. 参数校验
        if min_frames > max_frames:
            print(f"⚠️ 最小值({min_frames})>最大值({max_frames})，已设置min_frames=max_frames={max_frames}")
            min_frames = max_frames
        
        # 2. 解析输入
        try:
            segments = [int(x.strip()) for x in segment_frames.split(',') if x.strip()]
        except:
            segments = []
        
        if not segments:
            return ("", 0, 0, "", "")
        
        original_total = sum(segments)
        adjusted_segments = []
        carry_over = 0
        
        # 3. 处理所有片段
        for i, seg in enumerate(segments):
            current = seg + carry_over
            carry_over = 0
            is_last = (i == len(segments) - 1)
            
            # 关键修复：统一拆分条件
            while current > max_frames:
                # 拆分出max_frames片段
                adjusted_segments.append(max_frames)
                current -= max_frames
                
                # 如果是最后片段且剩余部分<min_frames，则合并
                if is_last and current < min_frames and adjusted_segments:
                    adjusted_segments[-1] += current
                    current = 0
                    break
            
            # 处理剩余部分
            if current > 0:
                if is_last and current < min_frames and adjusted_segments:
                    adjusted_segments[-1] += current
                else:
                    adjusted_segments.append(current)
        
        # 4. 后处理确保所有片段≥min_frames
        final_segments = []
        for seg in adjusted_segments:
            if seg >= min_frames:
                final_segments.append(seg)
            elif final_segments:
                final_segments[-1] += seg
            else:
                final_segments.append(seg)
        
        # 5. 计算总帧数和轮数
        total_frames = sum(final_segments)
        segment_count = len(final_segments)
        
        # 6. 最终验证
        if total_frames != original_total:
            if final_segments:
                final_segments[-1] += original_total - total_frames
            else:
                final_segments.append(original_total)
            total_frames = original_total
        
        # 7. 计算帧索引序列
        # segment_frames_index: 每个片段的起始帧索引
        segment_frames_index = []
        current_index = 0
        for seg in final_segments:
            segment_frames_index.append(str(current_index))
            current_index += seg
        
        # segment_frames_index_next: 每个片段的起始帧索引和起始帧索引+1
        segment_frames_index_next = []
        for idx in segment_frames_index:
            idx_int = int(idx)
            if idx_int<=0:
                continue
            segment_frames_index_next.extend([str(idx_int - 1),str(idx_int)])
        
        # 8. 转换为字符串
        adjusted_segments_str = ",".join(map(str, final_segments))
        segment_frames_index_str = ",".join(segment_frames_index)
        segment_frames_index_next_str = ",".join(segment_frames_index_next)
        
        return (
            adjusted_segments_str, 
            total_frames, 
            segment_count,
            segment_frames_index_str,
            segment_frames_index_next_str
        )
    
# 节点注册
NODE_CLASS_MAPPINGS = {
    "AnyBatchSliceNode": AnyBatchSliceNode,
    "SegmentFrameAdjuster":SegmentFrameAdjuster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyBatchSliceNode": "通用批处理切片",
    "SegmentFrameAdjuster": "通用每轮帧数限制",    
}

# 导出配置
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']