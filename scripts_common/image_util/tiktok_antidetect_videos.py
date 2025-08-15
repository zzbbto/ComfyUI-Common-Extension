import torch
import numpy as np
import math
import cv2
import random
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
from .tiktok_antidetect_utils import TikTokAntiDetectUtils
import comfy.utils  # 导入ComfyUI进度条工具


class TikTokAntiDetectVideo:
    """潜在空间的抖音AI检测对抗处理器 (2025.6优化版)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60, "tooltip": "视频帧率"}),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "启用抗检测处理"}),
                "overall_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1,
                                             "tooltip": "整体效果强度调节"}),
                "enable_camera_motion": ("BOOLEAN", {"default": True, "tooltip": "启用相机运动模拟"}),
                "camera_shake_intensity": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.1,
                                                   "tooltip": "相机抖动强度"}),
                "temporal_noise": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.05, "step": 0.001,
                                           "tooltip": "时域噪声强度"}),
                "enable_sensor_noise": ("BOOLEAN", {"default": True, "tooltip": "启用传感器噪声"}),
                "sensor_profile": (["iphone16", "pixel8", "xiaomi14", "action_cam"], 
                                  {"default": "xiaomi14", "tooltip": "手机传感器类型"}),
                "iso": ("INT", {"default": 800, "min": 100, "max": 3200, "step": 100,
                              "tooltip": "ISO感光度"}),
                "sensor_noise": ("FLOAT", {"default": 0.006, "min": 0.0, "max": 0.01, "step": 0.001,
                                          "tooltip": "传感器噪声强度"}),
                "enable_isp": ("BOOLEAN", {"default": True, "tooltip": "启用ISP色彩处理"}),
                "color_shift": ("FLOAT", {"default": 0.0015, "min": 0.0, "max": 0.01, "step": 0.0005,
                                         "tooltip": "色彩偏移强度"}),
                "enable_motion_blur": ("BOOLEAN", {"default": True, "tooltip": "启用运动模糊"}),
                "shutter_speed": ("FLOAT", {"default": 1/30, "min": 1/1000, "max": 1/10, "step": 1/1000,
                                           "tooltip": "快门速度(秒)"}),
                "motion_blur_strength": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1, "step": 0.1,
                                                 "tooltip": "运动模糊强度"}),
                "enable_compression": ("BOOLEAN", {"default": True, "tooltip": "启用压缩伪影"}),
                "tiktok_compression_level": (["low", "medium", "high"], {"default": "medium",
                                                                       "tooltip": "压缩等级"}),
                "compression_strength": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 0.05, "step": 0.005,
                                                  "tooltip": "压缩伪影强度"}),
                "output_for_pixel": ("BOOLEAN", {"default": True, "tooltip": "为像素处理优化输出"}),
                "physics_simulation": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.1,
                                               "tooltip": "物理引擎模拟强度"}),
                "quantum_protection": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1,
                                               "tooltip": "量子纠缠保护强度"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "process_video"
    CATEGORY = "video/postprocessing"
    OUTPUT_IS_LIST = (False,)

    def process_video(self, latent, fps, enable, overall_strength,
                    enable_camera_motion, camera_shake_intensity, temporal_noise,
                    enable_sensor_noise, sensor_profile, iso, sensor_noise,
                    enable_isp, color_shift,
                    enable_motion_blur, shutter_speed, motion_blur_strength,
                    enable_compression, tiktok_compression_level, compression_strength,
                    output_for_pixel, physics_simulation, quantum_protection):
        
        if not enable:
            return (latent,)
        
        samples = latent["samples"].clone()
        device = samples.device
        
        # 详细的形状检查和日志
        if samples.dim() == 4:
            batch_size, channels, height, width = samples.shape
            num_frames = batch_size
            # 验证通道数
            if channels == 0:
                print(f"⚠️ 警告：4D张量通道数为0! 形状: {samples.shape}")
                return (latent,)
                
        elif samples.dim() == 5:
            batch_size, time_size, channels, height, width = samples.shape
            num_frames = batch_size * time_size
            # 验证通道数
            if channels == 0:
                print(f"⚠️ 警告：5D张量通道数为0! 形状: {samples.shape}")
                return (latent,)
            # 添加形状验证
            expected_elements = batch_size * time_size * channels * height * width
            actual_elements = samples.numel()
            
            if expected_elements != actual_elements:
                print(f"⚠️ 形状不匹配! 预期元素: {expected_elements}, 实际元素: {actual_elements}")
                return (latent,)
            # 安全重塑
            samples = samples.reshape(-1, channels, height, width)
        else:
            print(f"⚠️ 不支持的张量维度: {samples.dim()}D, 形状: {samples.shape}")
            return (latent,)
        
        # 为后续像素处理调整强度
        if output_for_pixel:
            overall_strength = overall_strength * 0.7
            compression_strength = compression_strength * 0.5
        
        # 初始化进度条
        pbar = comfy.utils.ProgressBar(num_frames)
        pbar.update(0)
        
        camera_trajectory = TikTokAntiDetectUtils.generate_physical_trajectory(fps, num_frames, device)
        
        processed_frames = []
        prev_frame_samples = None
        
        for frame_idx in range(num_frames):
            frame_samples = samples[frame_idx:frame_idx+1].clone()
            
            # 2025.6新增：量子纠缠保护
            if quantum_protection > 0:
                frame_samples = TikTokAntiDetectUtils.apply_quantum_protection_latent(
                    frame_samples, quantum_protection, frame_idx, device
                )
            
            if enable_camera_motion:
                frame_samples = self.apply_camera_motion(
                    frame_samples, 
                    camera_trajectory[frame_idx], 
                    temporal_noise * overall_strength,
                    camera_shake_intensity,
                    device
                )
            
            if enable_sensor_noise:
                frame_samples = TikTokAntiDetectUtils.apply_sensor_noise(
                    frame_samples, 
                    sensor_noise * overall_strength, 
                    sensor_profile, 
                    iso, 
                    device,
                    in_latent_space=True
                )
            
            if enable_isp:
                frame_samples = TikTokAntiDetectUtils.apply_phone_isp(
                    frame_samples, 
                    color_shift * overall_strength, 
                    frame_idx, 
                    device,
                    in_latent_space=True
                )
            
            if enable_motion_blur:
                frame_samples = self.apply_motion_blur(
                    frame_samples, 
                    shutter_speed, 
                    motion_blur_strength * overall_strength,
                    device
                )
            
            # 2025.6新增：物理引擎模拟
            if physics_simulation > 0 and prev_frame_samples is not None:
                frame_samples = TikTokAntiDetectUtils.apply_physics_simulation(
                    frame_samples, prev_frame_samples, physics_simulation, device
                )
            
            if enable_compression:
                frame_samples = self.apply_tiktok_compression(
                    frame_samples, 
                    compression_strength * overall_strength, 
                    tiktok_compression_level, 
                    frame_idx, 
                    fps, 
                    device
                )
            
            processed_frames.append(frame_samples)
            prev_frame_samples = frame_samples.clone()
            
            # 更新进度条
            pbar.update(1)
        
        processed_samples = torch.cat(processed_frames, dim=0)
        
        if latent["samples"].dim() == 5:
            processed_samples = processed_samples.view(batch_size, time_size, channels, height, width)
        
        out_latent = latent.copy()
        out_latent["samples"] = processed_samples
        
        return (out_latent,)

    def apply_camera_motion(self, samples, pos, noise_strength, intensity, device):
        _, channels, height, width = samples.shape
        
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device), 
            torch.linspace(-1, 1, width, device=device), 
            indexing='ij'
        )
        
        dx = intensity * 0.03 * pos[0] * (1 - x_grid**2)
        dy = intensity * 0.03 * pos[1] * (1 - y_grid**2)
        
        samples = self.warp_image(samples, dx, dy)
        noise = torch.randn_like(samples) * noise_strength * 0.05
        return samples + noise

    def warp_image(self, img, dx, dy):
        batch, channels, height, width = img.shape
        device = img.device
        
        y, x = torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device), 
            indexing='ij'
        )
        x = x.float() / (width - 1) * 2 - 1
        y = y.float() / (height - 1) * 2 - 1
        
        x_new = torch.clamp(x + dx, -1, 1)
        y_new = torch.clamp(y + dy, -1, 1)
        
        grid = torch.stack((x_new, y_new), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
        return torch.nn.functional.grid_sample(
            img, grid, 
            align_corners=True, mode='bicubic', padding_mode='reflection'
        )

    def apply_motion_blur(self, frame_samples, shutter_speed, strength, device):
        if strength < 0.05:
            return frame_samples
            
        channels = frame_samples.shape[1]
        kernel_size = max(1, min(15, int(2 / (shutter_speed * 30))))
        
        if kernel_size < 3:
            return frame_samples
            
        kernel = torch.zeros(channels, 1, 1, kernel_size, device=device)
        kernel[:, :, :, :] = 1.0 / kernel_size
        
        blurred = torch.nn.functional.conv2d(
            frame_samples, kernel, 
            padding=(0, kernel_size//2),
            groups=channels
        )
        
        return frame_samples * (1 - strength) + blurred * strength

    def apply_tiktok_compression(self, samples, strength, level, frame_idx, fps, device):
        if samples.shape[1] < 3 or strength < 0.001:
            return samples
            
        rgb = samples[:, :3]
        original_rgb = rgb.clone().detach()
        other_channels = samples[:, 3:] if samples.shape[1] > 3 else None
        
        rgb_cpu = rgb.detach().cpu()
        rgb_np = rgb_cpu.mul(127.5).add(127.5).clamp(0, 255).byte().numpy()
        rgb_np = rgb_np.transpose(0, 2, 3, 1)
        
        processed_np = np.zeros_like(rgb_np)
        params = TikTokAntiDetectUtils.compression_params[level]
        
        for i in range(rgb_np.shape[0]):
            frame = rgb_np[i].copy()
            
            if strength > 0.001:
                frame = TikTokAntiDetectUtils.simulate_quantization(
                    frame, params["qp"], strength
                )
                frame = TikTokAntiDetectUtils.add_block_artifacts(
                    frame, params["bitrate_factor"], frame_idx, fps
                )
            
            if frame_idx > 0 and frame_idx % fps != 0:
                frame = TikTokAntiDetectUtils.add_motion_artifacts(
                    frame, params["bitrate_factor"]
                )
            
            processed_np[i] = frame
        
        processed_tensor = torch.from_numpy(processed_np.transpose(0, 3, 1, 2)).float()
        processed_tensor = processed_tensor.sub(127.5).div(127.5).to(device)
        
        processed_tensor = original_rgb * (1 - strength) + processed_tensor * strength
        
        if other_channels is not None:
            return torch.cat([processed_tensor, other_channels], dim=1)
        return processed_tensor

# # 节点映射
# NODE_CLASS_MAPPINGS = {
#     "TikTokAntiDetectVideo": TikTokAntiDetectVideo,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "TikTokAntiDetectVideo": "抖音AI检测对抗视频处理器",
# }