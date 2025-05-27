import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Any
from .base import BaseNode  # 假设ComfyUI的BaseNode路径

# 确保导入必要的类和函数
from .camera_control import CAM, CameraMotion, combine_poses, get_camera_motion, ndarray_to_poses

class SeductiveCameraGeneratorNode(BaseNode):
    """ComfyUI节点：生成美女诱惑主题的81帧随机相机运动"""
    
    # 美女诱惑主题常用相机运动组合（类型, 基础强度, 出现权重）
    SEDUCTIVE_MOTIONS = [
        ("TILT_UP", 0.8, 9),      # 上倾镜头，突出身材曲线
        ("TILT_DOWN", 0.7, 9),    # 下倾镜头，营造诱惑视角
        ("PAN_LEFT", 0.5, 7),     # 左平移，跟随动作
        ("PAN_RIGHT", 0.5, 7),    # 右平移，跟随动作
        ("ZOOM_IN", 0.9, 8),      # 拉近镜头，突出细节
        ("ZOOM_OUT", 0.6, 6),     # 拉远镜头，展示整体
        ("ROLL_CLOCKWISE", 0.4, 5), # 顺时针旋转，环绕效果
        ("ROLL_ANTICLOCKWISE", 0.4, 5), # 逆时针旋转，环绕效果
        ("STATIC", 0.0, 3)        # 静态帧，用于过渡
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """定义节点输入参数"""
        return {
            "required": {
                "motion_style": (["seductive", "dynamic", "gentle", "dramatic"], 
                               {"default": "seductive", "label": "镜头风格"}),
                "random_seed": ("INT", {"default": 42, "min": 0, "max": 999999, "step": 1}),
                "frame_count": ("INT", {"default": 81, "min": 1, "max": 200, "step": 1}),
                "motion_variation": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1, 
                                             "label": "运动变化程度"}),
                "speed_variation": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, 
                                             "label": "速度变化程度"}),
            },
            "optional": {
                "prev_poses": ("CAMERACTRL_POSES",),
            }
        }
    
    RETURN_TYPES = ("CAMERACTRL_POSES",)
    FUNCTION = "generate_seductive_poses"
    CATEGORY = "Animate Diff 🎭🅐🅓/③ Custom Nodes ③/Camera/Generators"
    
    def generate_random_motion_combination(self, n_motions=3, variation=0.8):
        """生成随机的相机运动组合"""
        if n_motions > 6:
            n_motions = 6
            
        # 根据运动权重随机选择
        motion_types = [item[0] for item in self.SEDUCTIVE_MOTIONS]
        motion_weights = [item[2] for item in self.SEDUCTIVE_MOTIONS]
        
        selected_indices = random.choices(
            range(len(self.SEDUCTIVE_MOTIONS)), 
            weights=motion_weights, 
            k=n_motions
        )
        
        motion_types_selected = []
        strengths = []
        
        for idx in selected_indices:
            motion_type = motion_types[idx]
            base_strength = self.SEDUCTIVE_MOTIONS[idx][1]
            # 添加随机强度变化，受variation参数影响
            strength = base_strength * random.uniform(0.8*variation, 1.2*variation)
            motion_types_selected.append(motion_type)
            strengths.append(strength)
        
        # 补充到6种运动（ComfyUI节点需要6个输入）
        while len(motion_types_selected) < 6:
            motion_types_selected.append("STATIC")
            strengths.append(0.0)
            
        return motion_types_selected, strengths
    
    def generate_style_based_speed(self, style_idx, variation=1.0):
        """根据镜头风格生成速度"""
        # 不同风格的基础速度范围
        style_speed_ranges = [
            (0.6, 1.2),   # seductive: 柔和诱人的速度
            (0.8, 1.5),   # dynamic: 动态速度
            (0.4, 0.9),   # gentle: 柔和速度
            (1.0, 2.0)    # dramatic: 剧烈变化的速度
        ]
        
        min_speed, max_speed = style_speed_ranges[style_idx]
        # 计算速度并添加方向变化
        base_speed = random.uniform(min_speed, max_speed) * variation
        direction = 1.0 if random.random() > 0.5 else -1.0
        return direction * base_speed
    
    def generate_segment_poses(self, motion_types, strengths, speed, frame_length):
        """生成单个运动片段的相机姿势"""
        # 组合相机运动
        combined_motion = CameraMotion.combine([
            CAM.get(motion_type).multiply(strength)
            for motion_type, strength in zip(motion_types, strengths)
        ])
        
        # 生成相机运动矩阵
        RT = get_camera_motion(
            combined_motion.rotate,
            combined_motion.translate,
            speed,
            n=frame_length
        )
        
        # 转换为相机姿势
        return ndarray_to_poses(RT)
    
    def generate_seductive_poses(self, 
                                 motion_style: str, 
                                 random_seed: int, 
                                 frame_count: int,
                                 motion_variation: float,
                                 speed_variation: float,
                                 prev_poses: list[list[float]] = None):
        """生成美女诱惑主题的相机运动序列"""
        # 设置随机种子保证可重复性
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 解析镜头风格
        style_map = {"seductive": 0, "dynamic": 1, "gentle": 2, "dramatic": 3}
        style_idx = style_map.get(motion_style, 0)
        
        all_poses = []
        prev_poses_ = prev_poses if prev_poses is not None else []
        segment_length = 12  # 每个片段的帧数
        segments = math.ceil(frame_count / segment_length)
        
        for segment_idx in range(segments):
            # 计算当前片段的实际帧数
            remaining_frames = frame_count - len(all_poses)
            if remaining_frames <= 0:
                break
                
            frame_length = min(segment_length, remaining_frames)
            
            # 生成随机运动组合
            n_motions = random.randint(2, 4)  # 每个片段选择2-4种运动
            motion_types, strengths = self.generate_random_motion_combination(
                n_motions=n_motions, 
                variation=motion_variation
            )
            
            # 生成适合风格的速度
            speed = self.generate_style_based_speed(
                style_idx=style_idx, 
                variation=speed_variation
            )
            
            # 生成当前片段的相机姿势
            segment_poses = self.generate_segment_poses(
                motion_types, strengths, speed, frame_length
            )
            
            # 组合到现有姿势中
            if prev_poses_:
                combined_poses = combine_poses(prev_poses_, segment_poses)
                all_poses.extend(combined_poses)
                prev_poses_ = combined_poses[-frame_length:]
            else:
                all_poses.extend(segment_poses)
                prev_poses_ = segment_poses
            
            # 达到所需帧数时停止
            if len(all_poses) >= frame_count:
                break
        
        # 确保生成指定数量的帧
        if len(all_poses) > frame_count:
            all_poses = all_poses[:frame_count]
        elif len(all_poses) < frame_count:
            # 不足时添加静态帧
            static_frames = frame_count - len(all_poses)
            static_poses = self.generate_static_poses(static_frames)
            if prev_poses_:
                all_poses = combine_poses(prev_poses_, static_poses)[:frame_count]
            else:
                all_poses.extend(static_poses[:static_frames])
        
        return (all_poses,)
    
    def generate_static_poses(self, frame_count=1):
        """生成静态相机姿势"""
        static_motion = CameraMotion(rotate=(0, 0, 0), translate=(0, 0, 0))
        RT = get_camera_motion(
            static_motion.rotate,
            static_motion.translate,
            speed=0,
            n=frame_count
        )
        return ndarray_to_poses(RT)

# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "SeductiveCameraGenerator": SeductiveCameraGeneratorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeductiveCameraGenerator": "美女诱惑相机生成器"
}