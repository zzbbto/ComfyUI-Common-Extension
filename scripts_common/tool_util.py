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
                "max_frames_all": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("每轮长度", "总帧数", "轮数", "每轮index", "每轮index和index+1")
    FUNCTION = "adjust_segments"
    CATEGORY = "CommonExtension/tool"

    def adjust_segments(self, segment_frames, max_frames, min_frames, max_frames_all):
        # 1. 参数校验
        if min_frames > max_frames:
            min_frames = max_frames
        
        # 2. 解析输入
        try:
            segments = [int(x.strip()) for x in segment_frames.split(',') if x.strip()]
        except:
            segments = []
        
        if not segments:
            return ("", 0, 0, "", "")
        
        original_total = sum(segments)
        total_limit = max_frames_all if max_frames_all > 0 else original_total
        
        # 3. 核心算法：帧片段调整
        adjusted_segments = []
        carry = 0
        total_used = 0
        
        for i, seg in enumerate(segments):
            if total_used >= total_limit:
                break
                
            # 计算当前可用帧数
            available = total_limit - total_used
            if available <= 0:
                break
                
            # 当前片段实际帧数（包括前一个片段的carry）
            current = min(seg, available) + carry
            carry = 0
            is_last = (i == len(segments) - 1)
            
            # 处理当前片段
            while current > 0 and total_used < total_limit:
                # 处理超过max_frames的情况
                if current > max_frames:
                    take = min(max_frames, total_limit - total_used)
                    adjusted_segments.append(take)
                    total_used += take
                    current -= take
                    
                    # 将多余帧移动到下一个片段（如果有）
                    if not is_last:
                        carry = current
                        break
                else:
                    # 处理最后片段不足min_frames的情况
                    if is_last and current < min_frames and adjusted_segments:
                        # 合并到前一个片段
                        adjusted_segments[-1] += min(current, total_limit - total_used)
                        total_used += min(current, total_limit - total_used)
                        current = 0
                    else:
                        # 直接添加片段
                        take = min(current, total_limit - total_used)
                        if take >= min_frames or (not adjusted_segments and take > 0):
                            adjusted_segments.append(take)
                            total_used += take
                            current -= take
                        elif adjusted_segments:
                            # 合并到前一个片段
                            adjusted_segments[-1] += min(current, total_limit - total_used)
                            total_used += min(current, total_used - total_used)
                            current = 0
                        else:
                            # 整个视频不足min_frames的特殊情况
                            adjusted_segments.append(take)
                            total_used += take
                            current = 0
        
        # 4. 最终调整确保片段有效
        final_segments = []
        for seg in adjusted_segments:
            if seg > 0:
                final_segments.append(seg)
        
        # 5. 确保最后片段满足min_frames要求
        if len(final_segments) > 1 and final_segments[-1] < min_frames:
            final_segments[-2] += final_segments[-1]
            final_segments.pop()
        
        # 6. 计算总帧数和片段数
        total_frames = sum(final_segments)
        segment_count = len(final_segments)
        
        # 7. 生成帧索引
        segment_frames_index = []
        segment_frames_index_next = []
        current_index = 0
        for seg in final_segments:
            segment_frames_index.append(str(current_index))
            segment_frames_index_next.extend([str(current_index), str(current_index + 1)])
            current_index += seg
        
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