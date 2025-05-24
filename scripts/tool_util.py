import math
from typing import Dict, Any
import comfy
from typing import Tuple
import torch

class RotationCameraCenterNode:
    
    # 常量定义（根据原始数据推导）
    PAN_BASE_INCREMENT = 0.09375  # 速度1时的X轴增量
    ZOOM_BASE_INCREMENT = 0.1875  # 速度1时的Z轴增量
    THRESHOLD = 0.5  # 速度切换阈值

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "frames_count": ("INT", {"default": 81, "min": 1}),
                "speed_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0}),
                "frame_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT")
    RETURN_NAMES = ("pan_direction", "pan_speed", "zoom_direction", "zoom_speed")
    FUNCTION = "calculate_rotation_params"
    CATEGORY = "CameraControl/tool"

    def calculate_rotation_params(
        self,
        frames_count: int = 81,
        speed_multiplier: float = 1.0,
        frame_index: int = 0
    ) -> Tuple[str, float, str, float]:
        """
        优化后的旋转参数计算逻辑
        - 使用极坐标分解确保路径闭合
        - 采用离散速度等级保持运动一致性
        - 增加参数范围校验
        """
        # 参数校验
        frame_index = max(0, min(frame_index, frames_count-1))
        speed_multiplier = max(0.1, min(speed_multiplier, 5.0))

        # 极坐标计算
        theta = (2 * math.pi * frame_index) / frames_count
        
        # X轴参数计算（Pan）
        pan_sign = math.cos(theta)  # 使用cos分量控制水平移动
        pan_direction = "Pan Right" if pan_sign >= 0 else "Pan Left"
        pan_speed = self._calculate_speed_level(abs(pan_sign)) * self.PAN_BASE_INCREMENT * speed_multiplier

        # Z轴参数计算（Zoom）
        zoom_sign = math.sin(theta)  # 使用sin分量控制纵深移动
        zoom_direction = "Zoom Out" if zoom_sign >= 0 else "Zoom In"
        zoom_speed = self._calculate_speed_level(abs(zoom_sign)) * self.ZOOM_BASE_INCREMENT * speed_multiplier

        return (pan_direction, abs(pan_speed),  # 保持速度绝对值为物理意义
                zoom_direction, abs(zoom_speed))

    def _calculate_speed_level(self, value: float) -> int:
        """离散速度转换逻辑"""
        return 2 if value >= self.THRESHOLD else 1

class AnyBatchSliceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_data": ("*", {}),      # 支持任意可切片类型
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

    RETURN_TYPES = ("*",)      # 保持原始数据类型
    RETURN_NAMES = ("sliced_data",)
    FUNCTION = "do_slice"
    CATEGORY = "CameraControl/tool"

    def do_slice(self, batch_data: Any, start_index: int, length: int):
        # 类型检查
        if not hasattr(batch_data, '__getitem__'):
            raise ValueError("Input data is not sliceable")

        # 处理负值逻辑
        max_length = len(batch_data) - start_index
        if length == -1:
            length = max_length
        else:
            length = min(length, max_length)

        # 边界检查
        if start_index < 0 or start_index >= len(batch_data):
            raise IndexError(f"Invalid start index: {start_index} (data length: {len(batch_data)})")
        
        if length <= 0:
            raise ValueError(f"Invalid length: {length}")

        # 执行切片操作
        end_index = start_index + length
        sliced = batch_data[start_index:end_index]

        # 特殊处理 Torch Tensor 的梯度
        if isinstance(sliced, torch.Tensor):
            sliced = sliced.detach().clone()

        return (sliced,)

NODE_CLASS_MAPPINGS = {
    "RotationCameraCenterNode": RotationCameraCenterNode,
    "AnyBatchSliceNode":AnyBatchSliceNode,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "RotationCameraCenterNode": "RotationCameraCenterNode",
    "AnyBatchSliceNode": "AnyBatchSliceNode",
    }