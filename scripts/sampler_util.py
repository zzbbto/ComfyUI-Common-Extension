import torch
import numpy as np
import math
import cv2
from tqdm import tqdm

class TikTokAntiDetectVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60, "tooltip": "视频帧率"}),
                
                # 总体控制
                "enable": ("BOOLEAN", {"default": True, "tooltip": "启用抗检测处理"}),
                "overall_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1,
                                             "tooltip": "整体效果强度调节"}),
                
                # 相机运动模块
                "enable_camera_motion": ("BOOLEAN", {"default": True, "tooltip": "启用相机运动模拟"}),
                "camera_shake_intensity": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.1,
                                                   "tooltip": "相机抖动强度"}),
                "temporal_noise": ("FLOAT", {"default": 0.008, "min": 0.0, "max": 0.02, "step": 0.001,
                                           "tooltip": "时域噪声强度"}),
                
                # 传感器噪声模块
                "enable_sensor_noise": ("BOOLEAN", {"default": True, "tooltip": "启用传感器噪声"}),
                "sensor_profile": (["iphone", "android_high", "android_mid", "action_cam"], 
                                  {"default": "iphone", "tooltip": "手机传感器类型"}),
                "iso": ("INT", {"default": 800, "min": 100, "max": 3200, "step": 100,
                              "tooltip": "ISO感光度"}),
                "sensor_noise": ("FLOAT", {"default": 0.004, "min": 0.0, "max": 0.01, "step": 0.001,
                                          "tooltip": "传感器噪声强度"}),
                
                # ISP色彩处理模块
                "enable_isp": ("BOOLEAN", {"default": True, "tooltip": "启用ISP色彩处理"}),
                "color_shift": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.003, "step": 0.0005,
                                         "tooltip": "色彩偏移强度"}),
                
                # 运动模糊模块
                "enable_motion_blur": ("BOOLEAN", {"default": True, "tooltip": "启用运动模糊"}),
                "shutter_speed": ("FLOAT", {"default": 1/30, "min": 1/1000, "max": 1/10, "step": 1/1000,
                                           "tooltip": "快门速度(秒)"}),
                "motion_blur_strength": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.5, "step": 0.1,
                                                 "tooltip": "运动模糊强度"}),
                
                # 压缩伪影模块
                "enable_compression": ("BOOLEAN", {"default": True, "tooltip": "启用压缩伪影"}),
                "tiktok_compression_level": (["low", "medium", "high"], {"default": "medium",
                                                                       "tooltip": "压缩等级"}),
                "compression_strength": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.025, "step": 0.005,
                                                  "tooltip": "压缩伪影强度"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "process_video"
    CATEGORY = "video/postprocessing"
    OUTPUT_IS_LIST = (False,)

    def __init__(self):
        self.sensor_params = {
            "iphone": {"read_noise": 2.1, "dark_current": 0.03, "gain": 1.8},
            "android_high": {"read_noise": 2.5, "dark_current": 0.05, "gain": 1.6},
            "android_mid": {"read_noise": 3.2, "dark_current": 0.08, "gain": 1.4},
            "action_cam": {"read_noise": 3.8, "dark_current": 0.12, "gain": 2.0}
        }
        self.compression_params = {
            "low": {"qp": 24, "bitrate_factor": 1.0},
            "medium": {"qp": 32, "bitrate_factor": 0.7},
            "high": {"qp": 40, "bitrate_factor": 0.5}
        }

    def process_video(self, latent, fps, enable, overall_strength,
                     enable_camera_motion, camera_shake_intensity, temporal_noise,
                     enable_sensor_noise, sensor_profile, iso, sensor_noise,
                     enable_isp, color_shift,
                     enable_motion_blur, shutter_speed, motion_blur_strength,
                     enable_compression, tiktok_compression_level, compression_strength):
        
        # 如果总开关关闭，直接返回原始latent
        if not enable:
            return (latent,)
        
        samples = latent["samples"].clone()
        device = samples.device
        
        # 获取视频尺寸信息
        batch_size, channels, height, width = samples.shape
        num_frames = batch_size
        
        # 生成相机轨迹 - 整个视频
        camera_trajectory = self.generate_physical_trajectory(fps, num_frames, device)
        
        # 处理整个视频
        processed_frames = []
        for frame_idx in tqdm(range(num_frames), desc="处理视频帧"):
            frame_samples = samples[frame_idx:frame_idx+1].clone()
            
            # 应用相机运动（轻微变形）
            if enable_camera_motion:
                frame_samples = self.apply_camera_motion(
                    frame_samples, 
                    camera_trajectory[frame_idx], 
                    temporal_noise * overall_strength,
                    camera_shake_intensity,
                    device
                )
            
            # 应用传感器噪声
            if enable_sensor_noise:
                frame_samples = self.apply_sensor_noise(
                    frame_samples, 
                    sensor_noise * overall_strength, 
                    sensor_profile, 
                    iso, 
                    device
                )
            
            # 应用ISP色彩处理
            if enable_isp:
                frame_samples = self.apply_phone_isp(
                    frame_samples, 
                    color_shift * overall_strength, 
                    frame_idx, 
                    device
                )
            
            # 应用运动模糊（仅在需要时启用）
            if enable_motion_blur:
                frame_samples = self.apply_motion_blur(
                    frame_samples, 
                    shutter_speed, 
                    motion_blur_strength * overall_strength,
                    device
                )
            
            # 应用抖音压缩效果（最后一步）
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
        
        # 组合所有处理后的帧
        processed_samples = torch.cat(processed_frames, dim=0)
        
        # 更新latent
        out_latent = latent.copy()
        out_latent["samples"] = processed_samples
        
        return (out_latent,)

    def generate_physical_trajectory(self, fps, num_frames, device):
        """生成自然的相机运动轨迹 - 整个视频"""
        # 基础运动 - 模拟手持拍摄的自然晃动
        t = torch.arange(num_frames, device=device) / fps
        
        # 多频率运动组合（更自然的轨迹）
        base_x = (
            0.15 * torch.sin(0.8 * t) + 
            0.08 * torch.sin(1.5 * t) +
            0.03 * torch.sin(0.3 * t)
        )
        
        base_y = (
            0.12 * torch.cos(0.7 * t) + 
            0.07 * torch.sin(1.2 * t) +
            0.04 * torch.cos(0.4 * t)
        )
        
        # 添加微小抖动
        rng = torch.Generator(device=device).manual_seed(42)
        jitter_x = torch.randn(num_frames, generator=rng, device=device) * 0.008
        jitter_y = torch.randn(num_frames, generator=rng, device=device) * 0.008
        
        return torch.stack((base_x + jitter_x, base_y + jitter_y), dim=1)

    def apply_camera_motion(self, samples, pos, noise_strength, intensity, device):
        """应用相机运动效果 - 轻微抖动（不导致模糊）"""
        _, channels, height, width = samples.shape
        
        # 创建网格坐标
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, height, device=samples.device), 
            torch.linspace(-1, 1, width, device=samples.device), 
            indexing='ij'
        )
        
        # 计算位移（使用更小的位移量）
        dx = intensity * 0.03 * pos[0] * (1 - x_grid**2)
        dy = intensity * 0.03 * pos[1] * (1 - y_grid**2)
        
        # 应用图像变形（轻微变形）
        samples = self.warp_image(samples, dx, dy)
        
        # 添加轻微的时域噪声
        noise = torch.randn_like(samples) * noise_strength * 0.05
        return samples + noise

    def warp_image(self, img, dx, dy):
        """图像变形函数 - 使用高质量插值"""
        batch, channels, height, width = img.shape
        device = img.device
        
        # 创建网格
        y, x = torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device), 
            indexing='ij'
        )
        x = x.float() / (width - 1) * 2 - 1
        y = y.float() / (height - 1) * 2 - 1
        
        # 应用位移（限制在合理范围内）
        x_new = torch.clamp(x + dx, -1, 1)
        y_new = torch.clamp(y + dy, -1, 1)
        
        # 创建变形网格（使用双三次插值保持清晰度）
        grid = torch.stack((x_new, y_new), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
        return torch.nn.functional.grid_sample(
            img, grid, 
            align_corners=True, 
            mode='bicubic',  # 使用双三次插值保持清晰度
            padding_mode='reflection'  # 反射填充避免边缘问题
        )

    def apply_sensor_noise(self, samples, strength, profile, iso, device):
        """添加传感器噪声 - 模拟真实手机摄像头噪声特性"""
        params = self.sensor_params[profile]
        
        # 计算不同类型的噪声
        read_noise = params["read_noise"] * (iso / 800) ** 0.5 * 0.01
        dark_current = params["dark_current"] * (iso / 800) * 0.01
        shot_noise = torch.sqrt(torch.abs(samples) * iso / 800) * 0.01
        
        # 组合总噪声
        total_noise = strength * (read_noise + dark_current + shot_noise)
        noise = torch.randn_like(samples) * total_noise
        return samples + noise

    def apply_phone_isp(self, samples, strength, frame_idx, device):
        """模拟手机ISP处理 - 轻微的色彩偏移"""
        if samples.shape[1] < 3:
            return samples
            
        # 创建ISP矩阵 - 轻微的调整
        isp_matrix = torch.tensor([
            [1.005, -0.002, -0.001],
            [-0.001, 1.008, -0.003],
            [0.001, -0.003, 0.998]
        ], device=samples.device).unsqueeze(0)
        
        # 随时间变化的自动白平衡偏移
        time_factor = torch.sin(torch.tensor(frame_idx * 0.03, device=samples.device)) * 0.5 + 0.5
        awb_shift = torch.tensor(
            [0.002 * time_factor, -0.001 * time_factor, -0.002 * time_factor], 
            device=samples.device
        ) * strength
        
        # 分离RGB通道
        rgb = samples[:, :3]
        other_channels = samples[:, 3:] if samples.shape[1] > 3 else None
        
        # 应用ISP变换
        batch, c, h, w = rgb.shape
        rgb = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb_transformed = torch.matmul(rgb, isp_matrix[0].t()) + awb_shift
        
        # 添加随机抖动
        jitter = torch.randn_like(rgb_transformed) * strength * 0.02
        rgb_transformed += jitter
        
        # 重组图像
        rgb_transformed = rgb_transformed.reshape(batch, h, w, c).permute(0, 3, 1, 2)
        
        if other_channels is not None:
            return torch.cat([rgb_transformed, other_channels], dim=1)
        else:
            return rgb_transformed

    def apply_motion_blur(self, frame_samples, shutter_speed, strength, device):
        """应用运动模糊 - 轻微效果（可选）"""
        # 只在需要时应用模糊
        if strength < 0.05:
            return frame_samples
            
        channels = frame_samples.shape[1]
        
        # 根据快门速度计算模糊程度
        kernel_size = max(1, min(15, int(2 / (shutter_speed * 30))))
        
        if kernel_size < 3:
            return frame_samples
            
        # 创建模糊核
        kernel = torch.zeros(channels, 1, 1, kernel_size, device=device)
        kernel[:, :, :, :] = 1.0 / kernel_size
        
        # 应用卷积模糊
        blurred = torch.nn.functional.conv2d(
            frame_samples, kernel, 
            padding=(0, kernel_size//2),
            groups=channels
        )
        
        # 轻微混合，避免过度模糊
        return frame_samples * (1 - strength) + blurred * strength

    def apply_tiktok_compression(self, samples, strength, level, frame_idx, fps, device):
        """模拟抖音压缩算法 - 添加压缩伪影（最后一步）"""
        if samples.shape[1] < 3 or strength < 0.001:
            return samples
            
        # 分离RGB通道
        rgb = samples[:, :3]
        
        # 保存原始RGB用于混合
        original_rgb = rgb.clone().detach()
        
        # 转换为CPU上的numpy数组进行处理
        rgb_cpu = rgb.detach().cpu()
        rgb_np = rgb_cpu.mul(127.5).add(127.5).clamp(0, 255).byte().numpy()
        rgb_np = rgb_np.transpose(0, 2, 3, 1)  # BCHW -> BHWC
        
        processed_np = np.zeros_like(rgb_np)
        
        for i in range(rgb_np.shape[0]):
            frame = rgb_np[i].copy()
            
            # 应用压缩效果
            if strength > 0.001:
                frame = self.simulate_quantization(frame, self.compression_params[level]["qp"], strength)
                frame = self.add_block_artifacts(frame, self.compression_params[level]["bitrate_factor"], frame_idx, fps)
            
            if frame_idx > 0 and frame_idx % fps != 0:
                frame = self.add_motion_artifacts(frame, self.compression_params[level]["bitrate_factor"])
            
            processed_np[i] = frame
        
        # 转换回tensor
        processed_tensor = torch.from_numpy(processed_np.transpose(0, 3, 1, 2)).float()
        processed_tensor = processed_tensor.sub(127.5).div(127.5).to(device)
        
        # 混合原始和处理后的图像
        processed_tensor = original_rgb * (1 - strength) + processed_tensor * strength
        
        # 重新组合通道
        if samples.shape[1] > 3:
            other_channels = samples[:, 3:]
            return torch.cat([processed_tensor, other_channels], dim=1)
        else:
            return processed_tensor

    def simulate_quantization(self, image, qp, strength):
        """模拟视频压缩的量化过程（优化性能）"""
        if image.size == 0 or strength < 0.001:
            return image
            
        # 转换为YUV色彩空间 - 模拟视频编码
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        # 仅在部分块上应用DCT变换和量化
        block_size = 8
        for y in range(0, yuv.shape[0], block_size):
            for x in range(0, yuv.shape[1], block_size):
                if y + block_size > yuv.shape[0] or x + block_size > yuv.shape[1]:
                    continue
                
                # 随机跳过一些块
                if np.random.random() > strength * 3:
                    continue
                
                for c in range(3):
                    # 提取块并应用DCT
                    block = yuv[y:y+block_size, x:x+block_size, c]
                    dct_block = cv2.dct(block.astype(np.float32))
                    
                    # 量化
                    quant_factor = max(1, qp / 31.0)
                    quant_block = np.round(dct_block / (16 * quant_factor))
                    dequant_block = quant_block * (16 * quant_factor)
                    
                    # 逆变换
                    idct_block = cv2.idct(dequant_block)
                    yuv[y:y+block_size, x:x+block_size, c] = np.clip(idct_block, 0, 255)
        
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    def add_block_artifacts(self, image, bitrate_factor, frame_idx, fps):
        """添加块状伪影 - 模拟低比特率压缩"""
        # 关键帧使用更大的块
        block_size = 16 if frame_idx % fps == 0 else 8
        
        grid_h = image.shape[0] // block_size
        grid_w = image.shape[1] // block_size
        
        for i in range(grid_h):
            for j in range(grid_w):
                # 随机添加伪影
                if np.random.random() > 0.3:
                    continue
                
                y_start = i * block_size
                x_start = j * block_size
                if y_start + block_size > image.shape[0] or x_start + block_size > image.shape[1]:
                    continue
                    
                # 轻微改变块内像素值
                block = image[y_start:y_start+block_size, x_start:x_start+block_size]
                factor = np.random.uniform(0.98, 1.02)
                image[y_start:y_start+block_size, x_start:x_start+block_size] = block * factor
        
        return image

    def add_motion_artifacts(self, image, bitrate_factor):
        """添加运动伪影 - 模拟运动补偿错误"""
        # 随机运动向量
        mv_x = np.random.randint(-1, 2)
        mv_y = np.random.randint(-1, 2)
        
        # 创建预测帧
        pred_frame = np.roll(image, mv_x, axis=1)
        pred_frame = np.roll(pred_frame, mv_y, axis=0)
        
        # 计算预测误差
        error = image - pred_frame
        
        # 混合原始和预测帧
        return image * 0.98 + pred_frame * 0.02 + error * 0.01 * bitrate_factor

# 节点映射
NODE_CLASS_MAPPINGS = {
    "TikTokAntiDetectVideo": TikTokAntiDetectVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TikTokAntiDetectVideo": "抖音AI检测对抗视频处理器",
}