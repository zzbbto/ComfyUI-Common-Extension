import torch
import numpy as np
import cv2
import random
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
from .tiktok_antidetect_utils import TikTokAntiDetectUtils
import comfy.utils  # 导入ComfyUI进度条工具


class TikTokAntiDetectPixel:
    """像素空间的抖音AI检测对抗处理器 (2025.6优化版)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60, "tooltip": "视频帧率"}),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "启用抗检测处理"}),
                "overall_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "整体效果强度调节"}),
                "enable_camera_motion": ("BOOLEAN", {"default": False, "tooltip": "启用相机运动模拟"}),
                "camera_shake_intensity": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.02,
                                                   "tooltip": "相机抖动强度"}),
                "temporal_noise": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 0.05, "step": 0.001,
                                           "tooltip": "时域噪声强度"}),
                "enable_sensor_noise": ("BOOLEAN", {"default": True, "tooltip": "启用传感器噪声"}),
                "sensor_profile": (["iphone16", "pixel8", "xiaomi14", "action_cam"], 
                                  {"default": "xiaomi14", "tooltip": "手机传感器类型"}),
                "iso": ("INT", {"default": 800, "min": 100, "max": 3200, "step": 100,
                              "tooltip": "ISO感光度"}),
                "sensor_noise": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.01, "step": 0.001,
                                          "tooltip": "传感器噪声强度"}),
                "enable_isp": ("BOOLEAN", {"default": True, "tooltip": "启用ISP色彩处理"}),
                "color_shift": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.01, "step": 0.0002,
                                         "tooltip": "色彩偏移强度"}),
                "enable_motion_blur": ("BOOLEAN", {"default": True, "tooltip": "启用运动模糊"}),
                "shutter_speed": ("FLOAT", {"default": 1/30, "min": 1/1000, "max": 1/10, "step": 1/1000,
                                           "tooltip": "快门速度(秒)"}),
                "motion_blur_strength": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1, "step": 0.05,
                                                 "tooltip": "运动模糊强度"}),
                "enable_compression": ("BOOLEAN", {"default": True, "tooltip": "启用压缩伪影"}),
                "tiktok_compression_level": (["low", "medium", "high"], {"default": "medium",
                                                                       "tooltip": "压缩等级"}),
                "compression_strength": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 0.05, "step": 0.003,
                                                  "tooltip": "压缩伪影强度"}),
                "preserve_original": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05,
                                               "tooltip": "保留原始图像比例"}),
                "is_secondary": ("BOOLEAN", {"default": False, "tooltip": "是否作为二级处理"}),
                "inject_fingerprint": ("BOOLEAN", {"default": True, "tooltip": "注入设备指纹"}),
                "device_model": (["iPhone16", "Pixel9", "Xiaomi14", "HuaweiP70"], 
                                {"default": "Xiaomi14", "tooltip": "模拟设备型号"}),
                "quantum_protection": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.1,
                                               "tooltip": "量子纠缠保护强度"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_images"
    CATEGORY = "image/postprocessing"
    OUTPUT_IS_LIST = (False,)

    def __init__(self):
        self.prev_frame = None
        self.prev_frame_info = {}
        self.fingerprint_cache = {}

    def process_images(self, images, fps, enable, overall_strength, preserve_original,
                     enable_camera_motion, camera_shake_intensity, temporal_noise,
                     enable_sensor_noise, sensor_profile, iso, sensor_noise,
                     enable_isp, color_shift,
                     enable_motion_blur, shutter_speed, motion_blur_strength,
                     enable_compression, tiktok_compression_level, compression_strength,
                     is_secondary, inject_fingerprint, device_model, quantum_protection):
        
        if not enable:
            return (images,)
        
        # 如果作为二级处理，调整参数
        if is_secondary:
            overall_strength = overall_strength * 0.8
            compression_strength = compression_strength * 1.2
        
        processed_images = images.clone().detach()
        device = processed_images.device
        
        # 获取图像尺寸信息
        num_frames, height, width, channels = processed_images.shape
        
        # 初始化进度条
        pbar = comfy.utils.ProgressBar(num_frames)
        pbar.update(0)
        
        # 生成设备指纹缓存
        if inject_fingerprint and device_model not in self.fingerprint_cache:
            self.fingerprint_cache[device_model] = self.generate_device_fingerprint(height, width, device_model)
        
        camera_trajectory = TikTokAntiDetectUtils.generate_physical_trajectory(fps, num_frames, device)
        
        for frame_idx in range(num_frames):
            img_tensor = processed_images[frame_idx:frame_idx+1]
            original_img = img_tensor.clone()
            
            # 2025.6新增：量子纠缠保护
            if quantum_protection > 0:
                img_tensor = TikTokAntiDetectUtils.apply_quantum_protection(
                    img_tensor, quantum_protection, frame_idx, device
                )
            
            if enable_camera_motion:
                img_tensor = self.apply_camera_motion(
                    img_tensor, 
                    camera_trajectory[frame_idx], 
                    temporal_noise * overall_strength,
                    camera_shake_intensity,
                    device
                )
            
            if enable_sensor_noise:
                img_tensor = TikTokAntiDetectUtils.apply_sensor_noise(
                    img_tensor, 
                    sensor_noise * overall_strength, 
                    sensor_profile, 
                    iso, 
                    device,
                    in_latent_space=False
                )
            
            if enable_isp:
                img_tensor = TikTokAntiDetectUtils.apply_phone_isp(
                    img_tensor, 
                    color_shift * overall_strength, 
                    frame_idx, 
                    device,
                    in_latent_space=False
                )
            
            if enable_motion_blur:
                img_tensor = self.apply_motion_blur(
                    img_tensor, 
                    shutter_speed, 
                    motion_blur_strength * overall_strength,
                    frame_idx,
                    device
                )
            
            if enable_compression:
                img_tensor = self.apply_tiktok_compression(
                    img_tensor, 
                    compression_strength * overall_strength, 
                    tiktok_compression_level, 
                    frame_idx, 
                    fps, 
                    device
                )
            
            # 2025.6新增：设备指纹注入
            if inject_fingerprint:
                fingerprint = self.fingerprint_cache[device_model]
                img_tensor = self.inject_device_fingerprint(img_tensor, fingerprint, device)
            
            processed_images[frame_idx] = original_img * preserve_original + img_tensor * (1 - preserve_original)
            
            self.prev_frame = img_tensor.clone().detach()
            self.prev_frame_info = {"frame_idx": frame_idx, "timestamp": frame_idx / fps}
            
            # 更新进度条
            pbar.update(1)
        
        return (processed_images,)

    def generate_device_fingerprint(self, height, width, device_model):
        """生成设备特定的传感器指纹"""
        # 创建基于设备型号的唯一种子
        seed = int(hashlib.sha256(device_model.encode()).hexdigest()[:8], 16) % (2**32)
        rng = np.random.default_rng(seed)
        
        # 生成光子级噪声模式
        fingerprint = rng.normal(loc=0.5, scale=0.02, size=(height, width, 3))
        
        # 添加设备特定的缺陷模式
        if "iPhone" in device_model:
            # iPhone特有的垂直条纹
            for i in range(0, width, 5):
                stripe_width = rng.integers(1, 3)
                fingerprint[:, i:i+stripe_width] += rng.normal(0, 0.01)
        
        elif "Xiaomi" in device_model:
            # 小米特有的网格模式
            grid_size = 8
            for i in range(0, height, grid_size):
                for j in range(0, width, grid_size):
                    if rng.random() > 0.7:
                        fingerprint[i:i+2, j:j+2] += rng.normal(0, 0.015)
        
        # 标准化到0-1范围
        fingerprint = np.clip(fingerprint, 0, 1)
        return torch.from_numpy(fingerprint).float()

    def inject_device_fingerprint(self, img_tensor, fingerprint, device):
        """将设备指纹注入到图像中"""
        batch, h, w, c = img_tensor.shape
        fingerprint = fingerprint.to(device)
        
        # 调整指纹尺寸以匹配图像
        if fingerprint.shape[0] != h or fingerprint.shape[1] != w:
            fingerprint = TF.resize(fingerprint.permute(2, 0, 1), (h, w)).permute(1, 2, 0)
        
        # 混合指纹到图像
        strength = 0.05  # 指纹强度
        return img_tensor * (1 - strength) + fingerprint * strength


    def apply_camera_motion(self, img_tensor, pos, noise_strength, intensity, device):
        batch, height, width, channels = img_tensor.shape
        
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device), 
            torch.linspace(-1, 1, width, device=device), 
            indexing='ij'
        )
        
        dx = intensity * 0.02 * pos[0] * (1 - x_grid**2)
        dy = intensity * 0.02 * pos[1] * (1 - y_grid**2)
        
        img_tensor = self.warp_image(img_tensor, dx, dy)
        noise = torch.randn_like(img_tensor) * noise_strength * 0.03
        return img_tensor + noise

    def warp_image(self, img, dx, dy):
        batch, height, width, channels = img.shape
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
        
        grid = torch.stack((x_new, y_new), dim=-1).unsqueeze(0)
        return torch.nn.functional.grid_sample(
            img.permute(0, 3, 1, 2), grid, 
            align_corners=True, mode='bicubic', padding_mode='reflection'
        ).permute(0, 2, 3, 1)

    def apply_motion_blur(self, img_tensor, shutter_speed, strength, frame_idx, device):
        if strength < 0.03:
            return img_tensor
        
        if frame_idx == 0 or self.prev_frame is None:
            return self.apply_simple_blur(img_tensor, strength, device)
        
        motion_vector = self.calculate_motion_vector(img_tensor, self.prev_frame)
        kernel_size = max(1, min(15, int(1.5 / (shutter_speed * 30))))
        return self.apply_directional_blur(img_tensor, motion_vector, kernel_size, strength, device)

    def calculate_motion_vector(self, current_frame, prev_frame):
        current_gray = current_frame.mean(dim=-1, keepdim=True)
        prev_gray = prev_frame.mean(dim=-1, keepdim=True)
        diff = current_gray - prev_gray
        
        motion_x = diff[:, :, 1:, :] - diff[:, :, :-1, :]
        motion_y = diff[:, 1:, :, :] - diff[:, :-1, :, :]
        
        avg_motion_x = motion_x.mean().item()
        avg_motion_y = motion_y.mean().item()
        
        max_motion = 0.1
        avg_motion_x = max(-max_motion, min(max_motion, avg_motion_x))
        avg_motion_y = max(-max_motion, min(max_motion, avg_motion_y))
        
        return (avg_motion_x, avg_motion_y)

    def apply_directional_blur(self, img_tensor, motion_vector, kernel_size, strength, device):
        kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=device)
        center = kernel_size // 2
        
        dx, dy = motion_vector
        length = max(1, min(kernel_size//2, int(kernel_size * abs(dx + dy) * 5)))
        
        if abs(dx) > abs(dy):
            kernel[0, 0, center, center-length:center+length+1] = 1.0 / (2*length+1)
        else:
            kernel[0, 0, center-length:center+length+1, center] = 1.0 / (2*length+1)
        
        blurred = torch.nn.functional.conv2d(
            img_tensor.permute(0, 3, 1, 2), 
            kernel.repeat(3, 1, 1, 1), 
            padding=center,
            groups=3
        ).permute(0, 2, 3, 1)
        
        return img_tensor * (1 - strength) + blurred * strength

    def apply_simple_blur(self, img_tensor, strength, device):
        blurred = TF.gaussian_blur(
            img_tensor.permute(0, 3, 1, 2), 
            kernel_size=3,
            sigma=0.5 + strength
        ).permute(0, 2, 3, 1)
        return img_tensor * (1 - strength) + blurred * strength

    def apply_tiktok_compression(self, img_tensor, strength, level, frame_idx, fps, device):
        if img_tensor.shape[-1] < 3 or strength < 0.001:
            return img_tensor
        
        original_img = img_tensor.clone().detach()
        img_cpu = img_tensor.detach().cpu().numpy()
        img_np = (img_cpu * 255).clip(0, 255).astype(np.uint8)
        
        processed_np = np.zeros_like(img_np)
        params = TikTokAntiDetectUtils.compression_params[level]
        
        for i in range(img_np.shape[0]):
            frame = img_np[i].copy()
            
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
        
        processed_tensor = torch.from_numpy(processed_np / 255.0).float().to(device)
        return original_img * (1 - strength) + processed_tensor * strength

# 节点映射
NODE_CLASS_MAPPINGS = {
    "TikTokAntiDetectPixel": TikTokAntiDetectPixel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TikTokAntiDetectPixel": "抖音AI检测对抗像素处理器",
}