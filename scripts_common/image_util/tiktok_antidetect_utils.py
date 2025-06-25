import torch
import numpy as np
import cv2
import random
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
import hashlib

class TikTokAntiDetectUtils:
    """抖音AI检测对抗的通用工具类 (2025.6优化版)"""
    # 传感器参数更新为2025年设备
    sensor_params = {
        "iphone16": {"read_noise": 1.8, "dark_current": 0.025, "gain": 2.0},
        "pixel8": {"read_noise": 2.0, "dark_current": 0.03, "gain": 1.9},
        "xiaomi14": {"read_noise": 2.2, "dark_current": 0.035, "gain": 1.8},
        "action_cam": {"read_noise": 3.0, "dark_current": 0.10, "gain": 2.2},
        "dslr": {"read_noise": 1.5, "dark_current": 0.015, "gain": 1.0}
    }
    
    # 压缩参数更新为2025年标准
    compression_params = {
        "low": {"qp": 28, "bitrate_factor": 1.2},
        "medium": {"qp": 36, "bitrate_factor": 0.8},
        "high": {"qp": 44, "bitrate_factor": 0.6}
    }
    
    # 物理模拟参数
    physics_params = {
        "gravity": 9.80665,
        "light_models": {
            "daylight": {"intensity": 1.0, "temperature": 5500},
            "cloudy": {"intensity": 0.8, "temperature": 6500},
            "indoor": {"intensity": 0.7, "temperature": 4000},
            "night": {"intensity": 0.3, "temperature": 3000}
        }
    }
    
    @staticmethod
    def generate_physical_trajectory(fps, num_frames, device):
        """生成更自然的相机运动轨迹 (2025.6优化)"""
        t = torch.arange(num_frames, device=device) / fps
        
        # 多频率运动组合 - 更复杂的模式
        base_x = (
            0.15 * torch.sin(0.65 * t) + 
            0.07 * torch.sin(1.4 * t + 0.5) +
            0.03 * torch.sin(0.22 * t) +
            0.02 * torch.cos(2.1 * t)
        )
        
        base_y = (
            0.12 * torch.cos(0.55 * t) + 
            0.05 * torch.sin(1.2 * t - 0.3) +
            0.04 * torch.cos(0.3 * t) +
            0.01 * torch.sin(1.8 * t)
        )
        
        # 添加符合物理规律的抖动
        rng = torch.Generator(device=device).manual_seed(42)
        jitter_x = torch.randn(num_frames, generator=rng, device=device) * 0.004
        jitter_y = torch.randn(num_frames, generator=rng, device=device) * 0.004
        
        return torch.stack((base_x + jitter_x, base_y + jitter_y), dim=1)

    @staticmethod
    def apply_quantum_protection(data, strength, frame_idx, device):
        """应用量子纠缠保护 (像素空间) - 修复版"""
        # 获取图像尺寸
        h, w = data.shape[1:3] if data.dim() == 4 else data.shape[:2]
        
        # 创建适合图像尺寸的量子签名
        signature = TikTokAntiDetectUtils.generate_quantum_signature(frame_idx, h, w)
        
        # 应用保护
        protected = data.clone()
        for c in range(3):
            channel_data = data[..., c]
            channel_data_np = channel_data.cpu().numpy()
            
            # 应用FFT
            fft_channel = np.fft.fft2(channel_data_np)
            
            # 应用量子签名
            protected_fft = fft_channel * signature
            
            # 逆FFT
            quantum_layer = np.abs(np.fft.ifft2(protected_fft))
            quantum_layer_tensor = torch.from_numpy(quantum_layer).float().to(device)
            
            # 混合结果
            protected[..., c] = channel_data * (1 - strength) + quantum_layer_tensor * strength
        
        return protected

    @staticmethod
    def apply_quantum_protection_latent(samples, strength, frame_idx, device):
        """应用量子纠缠保护 (潜在空间) - 修复版"""
        # 获取潜在空间尺寸
        _, _, h, w = samples.shape
        
        # 创建适合潜在空间尺寸的量子签名
        signature = TikTokAntiDetectUtils.generate_quantum_signature(frame_idx, h, w)
        signature_tensor = torch.from_numpy(signature).float().to(device)
        
        # 只处理前3个通道 (RGB)
        if samples.shape[1] >= 3:
            rgb = samples[:, :3]
            
            # 应用量子保护
            protected_rgb = rgb.clone()
            for i in range(3):
                channel = rgb[:, i:i+1]
                
                # 应用FFT
                fft_channel = torch.fft.fft2(channel)
                
                # 应用量子签名
                protected_fft = fft_channel * signature_tensor
                
                # 逆FFT
                protected_channel = torch.fft.ifft2(protected_fft).real
                
                # 混合结果
                protected_rgb[:, i:i+1] = channel * (1 - strength) + protected_channel * strength
            
            # 组合回原始样本
            if samples.shape[1] > 3:
                return torch.cat([protected_rgb, samples[:, 3:]], dim=1)
            return protected_rgb
        
        return samples

    @staticmethod
    def generate_quantum_signature(frame_idx, height=8, width=8):
        """生成量子签名 (动态尺寸)"""
        seed = (frame_idx * 137) % 256
        rng = np.random.default_rng(seed)
        
        # 创建频率域签名 (使用实际图像尺寸)
        signature = np.ones((height, width), dtype=np.complex64)
        
        # 仅在中心区域添加变化 (避免边缘问题)
        center_h, center_w = height // 2, width // 2
        block_size = min(8, height, width)  # 确保不超过图像尺寸
        
        start_h = max(0, center_h - block_size // 2)
        end_h = min(height, start_h + block_size)
        start_w = max(0, center_w - block_size // 2)
        end_w = min(width, start_w + block_size)
        
        for i in range(start_h, end_h):
            for j in range(start_w, end_w):
                if rng.random() > 0.7:
                    phase = rng.uniform(0, 2 * np.pi)
                    magnitude = rng.uniform(0.8, 1.2)
                    signature[i, j] = magnitude * (np.cos(phase) + 1j * np.sin(phase))
        
        return signature

    @staticmethod
    def apply_physics_simulation(current_frame, prev_frame, strength, device):
        """应用物理引擎模拟 (2025.6新增)"""
        # 计算运动向量
        motion_vector = TikTokAntiDetectUtils.calculate_motion_vector(
            current_frame, prev_frame, device
        )
        
        # 应用物理校正
        batch, channels, height, width = current_frame.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing='ij'
        )
        
        # 基于运动向量创建扭曲场
        dx = motion_vector[0] * strength * 0.1 * (1 - grid_x**2)
        dy = motion_vector[1] * strength * 0.1 * (1 - grid_y**2)
        
        # 应用扭曲
        return TikTokAntiDetectUtils.warp_image(current_frame, dx, dy, device)

    @staticmethod
    def calculate_motion_vector(current_frame, prev_frame, device):
        """计算帧间运动向量 (物理模拟用)"""
        if current_frame.shape[1] < 3 or prev_frame.shape[1] < 3:
            return (0.0, 0.0)
        
        # 提取亮度通道
        current_gray = current_frame[:, :3].mean(dim=1, keepdim=True)
        prev_gray = prev_frame[:, :3].mean(dim=1, keepdim=True)
        
        # 计算差异
        diff = current_gray - prev_gray
        
        # 计算X和Y方向的梯度
        grad_x = diff[:, :, :, 1:] - diff[:, :, :, :-1]
        grad_y = diff[:, :, 1:, :] - diff[:, :, :-1, :]
        
        # 计算平均运动
        avg_motion_x = grad_x.mean().item()
        avg_motion_y = grad_y.mean().item()
        
        # 限制最大运动范围
        max_motion = 0.15
        avg_motion_x = max(-max_motion, min(max_motion, avg_motion_x))
        avg_motion_y = max(-max_motion, min(max_motion, avg_motion_y))
        
        return (avg_motion_x, avg_motion_y)

    @staticmethod
    def warp_image(img, dx, dy, device):
        """扭曲图像 (物理模拟用)"""
        batch, channels, height, width = img.shape
        
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

    @staticmethod
    def apply_sensor_noise(data, strength, profile, iso, device, in_latent_space):
        """添加传感器噪声 - 通用版本"""
        params = TikTokAntiDetectUtils.sensor_params[profile]
        
        read_noise = params["read_noise"] * (iso / 800) ** 0.5 * 0.01
        dark_current = params["dark_current"] * (iso / 800) * 0.01
        
        if in_latent_space:
            shot_noise = torch.sqrt(torch.abs(data) * iso / 800) * 0.01
        else:
            # 像素空间的噪声计算更精细
            shot_noise = torch.sqrt(torch.clamp(data, 0.01, 1.0) * iso / 800) * 0.01
        
        total_noise = strength * (read_noise + dark_current + shot_noise)
        noise = torch.randn_like(data) * total_noise
        return data + noise

    @staticmethod
    def apply_phone_isp(data, strength, frame_idx, device, in_latent_space):
        """模拟手机ISP处理 - 通用版本"""
        # 随时间变化的自动白平衡偏移
        time_factor = torch.sin(torch.tensor(frame_idx * 0.02, device=device)) * 0.5 + 0.5
        awb_shift = torch.tensor(
            [0.001 * time_factor, -0.0005 * time_factor, -0.001 * time_factor], 
            device=device
        ) * strength
        
        if in_latent_space:
            # Latent空间处理 (B, C, H, W)
            if data.shape[1] < 3:
                return data
                
            # 创建ISP矩阵 - 轻微的调整
            isp_matrix = torch.tensor([
                [1.005, -0.002, -0.001],
                [-0.001, 1.008, -0.003],
                [0.001, -0.003, 0.998]
            ], device=device).unsqueeze(0)
            
            rgb = data[:, :3]
            other_channels = data[:, 3:] if data.shape[1] > 3 else None
            
            batch, c, h, w = rgb.shape
            rgb = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
            rgb_transformed = torch.matmul(rgb, isp_matrix[0].t()) + awb_shift
            
            jitter = torch.randn_like(rgb_transformed) * strength * 0.02
            rgb_transformed += jitter
            
            rgb_transformed = rgb_transformed.reshape(batch, h, w, c).permute(0, 3, 1, 2)
            
            if other_channels is not None:
                return torch.cat([rgb_transformed, other_channels], dim=1)
            return rgb_transformed
        else:
            # 像素空间处理 (B, H, W, C)
            if data.shape[-1] < 3:
                return data
                
            # 创建ISP矩阵 - 轻微的调整
            isp_matrix = torch.tensor([
                [1.004, -0.001, -0.001],
                [-0.001, 1.006, -0.002],
                [0.001, -0.002, 0.997]
            ], device=device).unsqueeze(0)
            
            rgb = data[..., :3]
            other_channels = data[..., 3:] if data.shape[-1] > 3 else None
            
            batch, h, w, c = data.shape
            rgb_flat = rgb.reshape(-1, 3)
            rgb_transformed = torch.matmul(rgb_flat, isp_matrix[0].t()) + awb_shift
            
            jitter = torch.randn_like(rgb_transformed) * strength * 0.01
            rgb_transformed += jitter
            
            rgb_transformed = rgb_transformed.reshape(batch, h, w, c)
            
            if other_channels is not None:
                return torch.cat([rgb_transformed, other_channels], dim=-1)
            return rgb_transformed

    @staticmethod
    def simulate_quantization(image, qp, strength):
        """模拟视频压缩的量化过程 - 通用版本"""
        if image.size == 0 or strength < 0.001:
            return image
            
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        height, width = yuv.shape[:2]
        block_size = 8
        
        # 随机选择要处理的块
        block_coords = [(y, x) for y in range(0, height, block_size) 
                       for x in range(0, width, block_size)]
        random.shuffle(block_coords)
        num_blocks_to_process = int(len(block_coords) * strength * 0.8)
        
        for idx in range(num_blocks_to_process):
            y, x = block_coords[idx]
            if y + block_size > height or x + block_size > width:
                continue
                
            for c in range(3):
                block = yuv[y:y+block_size, x:x+block_size, c]
                dct_block = cv2.dct(block.astype(np.float32))
                
                quant_factor = max(1, qp / 31.0)
                quant_block = np.round(dct_block / (16 * quant_factor))
                dequant_block = quant_block * (16 * quant_factor)
                
                idct_block = cv2.idct(dequant_block)
                yuv[y:y+block_size, x:x+block_size, c] = np.clip(idct_block, 0, 255)
        
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    @staticmethod
    def add_block_artifacts(image, bitrate_factor, frame_idx, fps):
        """添加块状伪影 - 通用版本"""
        block_size = 16 if frame_idx % fps == 0 else 8
        grid_h = image.shape[0] // block_size
        grid_w = image.shape[1] // block_size
        
        for i in range(grid_h):
            for j in range(grid_w):
                if random.random() > 0.4 * bitrate_factor:
                    continue
                
                y_start = i * block_size
                x_start = j * block_size
                if y_start + block_size > image.shape[0] or x_start + block_size > image.shape[1]:
                    continue
                    
                block = image[y_start:y_start+block_size, x_start:x_start+block_size]
                
                if block_size > 4:
                    inner = block[1:-1, 1:-1]
                    inner[:] = gaussian_filter(inner, sigma=0.7)
                    
                    block[0, :] = block[0, :] * 1.05
                    block[-1, :] = block[-1, :] * 1.05
                    block[:, 0] = block[:, 0] * 1.05
                    block[:, -1] = block[:, -1] * 1.05
                
                color_shift = random.uniform(0.98, 1.02)
                image[y_start:y_start+block_size, x_start:x_start+block_size] = block * color_shift
        
        return image

    @staticmethod
    def add_motion_artifacts(image, bitrate_factor):
        """添加运动伪影 - 通用版本"""
        mv_x = random.randint(-1, 1)
        mv_y = random.randint(-1, 1)
        
        pred_frame = np.roll(image, mv_x, axis=1)
        pred_frame = np.roll(pred_frame, mv_y, axis=0)
        
        error = image - pred_frame
        return image * 0.99 + pred_frame * 0.01 + error * 0.005 * bitrate_factor