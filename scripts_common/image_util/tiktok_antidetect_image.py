import torch
import numpy as np
import cv2
import random
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
import comfy.utils
import hashlib
import math
from .tiktok_antidetect_utils import TikTokAntiDetectUtils

class ZhuqueAntiDetectEnhancer:
    """朱雀AI检测专用增强处理器 (2025.8优化版) - 色彩优化版"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "启用抗检测处理"}),
                "overall_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "整体效果强度调节"}),
                "zhuque_bypass_mode": (["balanced", "aggressive", "stealth"], 
                                      {"default": "balanced", "tooltip": "朱雀AI绕过模式"}),
                "sensor_profile": (["iphone16", "pixel8", "xiaomi14", "action_cam", "dslr"], 
                                  {"default": "dslr", "tooltip": "传感器类型"}),
                "iso": ("INT", {"default": 400, "min": 100, "max": 6400, "step": 100,
                              "tooltip": "ISO感光度"}),
                
                # 量子保护模块控制
                "enable_quantum_protection": ("BOOLEAN", {"default": True, "tooltip": "启用量子纠缠保护"}),
                "quantum_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "量子保护强度"}),
                
                # 镜头效果模块控制
                "enable_lens_flare": ("BOOLEAN", {"default": True, "tooltip": "启用镜头炫光"}),
                "lens_flare_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.3, "step": 0.05,
                                                "tooltip": "镜头炫光强度"}),
                "enable_chromatic_aberration": ("BOOLEAN", {"default": True, "tooltip": "启用色差"}),
                "chromatic_aberration_strength": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.2, "step": 0.05,
                                                         "tooltip": "色差强度"}),
                
                # 传感器模块控制
                "enable_sensor_defects": ("BOOLEAN", {"default": True, "tooltip": "启用传感器缺陷"}),
                "sensor_defects_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05,
                                                    "tooltip": "传感器缺陷强度"}),
                "enable_sensor_noise": ("BOOLEAN", {"default": True, "tooltip": "启用传感器噪声"}),
                "sensor_noise_strength": ("FLOAT", {"default": 0.004, "min": 0.0, "max": 0.008, "step": 0.001,
                                                  "tooltip": "传感器噪声强度"}),
                
                # 动态范围模块控制
                "enable_dynamic_range": ("BOOLEAN", {"default": True, "tooltip": "启用动态范围压缩"}),
                "dynamic_range_strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.15, "step": 0.05,
                                                   "tooltip": "动态范围压缩强度"}),
                
                # 压缩模块控制
                "enable_h265_compression": ("BOOLEAN", {"default": True, "tooltip": "启用H.265压缩模拟"}),
                "h265_compression_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.4, "step": 0.05,
                                                     "tooltip": "H.265压缩伪影强度"}),
                
                # 色彩响应模块控制
                "enable_color_response": ("BOOLEAN", {"default": True, "tooltip": "启用设备色彩响应"}),
                "color_response_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1,
                                                    "tooltip": "色彩响应强度"}),
                
                # EXIF元数据模块控制
                "enable_exif_metadata": ("BOOLEAN", {"default": True, "tooltip": "启用EXIF元数据注入"}),
                "device_model": (["Canon_EOS_R5", "Sony_A7IV", "iPhone16_Pro", "Xiaomi14_Ultra"], 
                                {"default": "Sony_A7IV", "tooltip": "模拟设备型号"}),
                "brightness_correction": ("FLOAT", {"default": 1.05, "min": 0.9, "max": 1.2, "step": 0.01,
                                                  "tooltip": "亮度校正因子"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "enhance_images"
    CATEGORY = "image/postprocessing"
    OUTPUT_IS_LIST = (False,)

    def __init__(self):
        self.device_profiles = {
            "Canon_EOS_R5": {
                "sensor_noise": 1.0,
                "color_response": [1.02, 1.00, 0.98],  # 更明亮的色彩响应
                "defect_pattern": "grid",
                "dynamic_range": 14.5
            },
            "Sony_A7IV": {
                "sensor_noise": 0.7,
                "color_response": [1.04, 0.99, 1.01],  # 轻微提升红色
                "defect_pattern": "random",
                "dynamic_range": 15.0
            },
            "iPhone16_Pro": {
                "sensor_noise": 2.0,
                "color_response": [1.00, 1.03, 0.97],  # 提升绿色通道
                "defect_pattern": "vertical",
                "dynamic_range": 12.0
            },
            "Xiaomi14_Ultra": {
                "sensor_noise": 2.5,
                "color_response": [1.01, 0.98, 1.02],  # 平衡的响应曲线
                "defect_pattern": "circular",
                "dynamic_range": 12.5
            }
        }
        self.zhuque_modes = {
            "balanced": {"noise_boost": 1.0, "defect_strength": 0.8, "compression_factor": 1.0},
            "aggressive": {"noise_boost": 1.2, "defect_strength": 1.0, "compression_factor": 1.2},
            "stealth": {"noise_boost": 0.8, "defect_strength": 0.6, "compression_factor": 0.8}
        }

    def enhance_images(self, images, enable, overall_strength, zhuque_bypass_mode,
                      sensor_profile, iso, 
                      enable_quantum_protection, quantum_strength,
                      enable_lens_flare, lens_flare_strength,
                      enable_chromatic_aberration, chromatic_aberration_strength,
                      enable_sensor_defects, sensor_defects_strength,
                      enable_sensor_noise, sensor_noise_strength,
                      enable_dynamic_range, dynamic_range_strength,
                      enable_h265_compression, h265_compression_strength,
                      enable_color_response, color_response_strength,
                      enable_exif_metadata, device_model, brightness_correction):
        
        if not enable:
            return (images,)
        
        processed_images = images.clone().detach()
        device = processed_images.device
        
        # 获取朱雀模式参数
        mode_params = self.zhuque_modes[zhuque_bypass_mode]
        
        # 获取设备参数
        device_params = self.device_profiles[device_model]
        
        # 获取批次信息
        batch_size, height, width, channels = processed_images.shape
        
        # 初始化进度条
        pbar = comfy.utils.ProgressBar(batch_size)
        
        # 预生成传感器缺陷图
        defect_pattern = self.generate_sensor_defect_pattern(height, width, device_params["defect_pattern"])
        defect_pattern = defect_pattern.to(device)
        
        # 处理每张图片
        for i in range(batch_size):
            img_tensor = processed_images[i:i+1]
            
            # 应用传感器噪声和缺陷
            if enable_sensor_noise or enable_sensor_defects:
                img_tensor = self.apply_sensor_characteristics(
                    img_tensor, 
                    sensor_profile,
                    iso,
                    device_params["sensor_noise"] * mode_params["noise_boost"],
                    defect_pattern,
                    sensor_defects_strength * mode_params["defect_strength"],
                    sensor_noise_strength,
                    device,
                    enable_sensor_noise,
                    enable_sensor_defects
                )
            
            # 量子纠缠保护
            if enable_quantum_protection:
                img_tensor = self.apply_quantum_protection(
                    img_tensor, 
                    quantum_strength * overall_strength, 
                    random.randint(0, 100000),  # 唯一帧ID
                    device
                )
            
            # 镜头光学效果
            if enable_lens_flare:
                img_tensor = self.add_lens_flare(img_tensor, lens_flare_strength, device)
            
            if enable_chromatic_aberration:
                img_tensor = self.add_chromatic_aberration(img_tensor, chromatic_aberration_strength, device)
            
            # 动态范围压缩 (减轻效果)
            if enable_dynamic_range:
                img_tensor = self.compress_dynamic_range(
                    img_tensor,
                    dynamic_range_strength,
                    device_params["dynamic_range"],
                    device
                )
            
            # H.265压缩伪影
            if enable_h265_compression:
                img_tensor = self.apply_h265_compression(
                    img_tensor,
                    h265_compression_strength * mode_params["compression_factor"],
                    device
                )
            
            # 设备色彩响应 (减轻效果)
            if enable_color_response:
                img_tensor = self.apply_color_response(
                    img_tensor,
                    device_params["color_response"],
                    color_response_strength,
                    device
                )
            
            # 应用亮度校正 (新增)
            img_tensor = img_tensor * brightness_correction
            
            # 确保像素值在[0,1]范围内
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
            processed_images[i] = img_tensor[0]
            pbar.update(1)
        
        # EXIF元数据注入
        if enable_exif_metadata:
            processed_images = self.inject_exif_metadata(processed_images, device_model, iso)
        
        return (processed_images,)

    def generate_sensor_defect_pattern(self, height, width, pattern_type):
        """生成传感器缺陷模式 (减轻效果)"""
        pattern = torch.zeros((height, width, 3))
        
        if pattern_type == "grid":
            # 更轻微的网格缺陷
            for i in range(0, height, 10):  # 增加网格间距
                for j in range(0, width, 10):
                    if random.random() > 0.8:  # 减少出现概率
                        intensity = random.uniform(0.003, 0.008)  # 降低强度
                        pattern[i:i+1, j:j+1] += intensity
        
        elif pattern_type == "vertical":
            # 更轻微的垂直线缺陷
            for j in range(0, width, 6):  # 增加线间距
                if random.random() > 0.85:  # 减少出现概率
                    stripe_width = random.randint(1, 2)  # 更窄的条纹
                    intensity = random.uniform(0.003, 0.007)  # 降低强度
                    pattern[:, j:j+stripe_width] += intensity
        
        elif pattern_type == "circular":
            # 更少的热像素
            for _ in range(int(height * width * 0.00003)):  # 进一步减少热像素数量
                y = random.randint(0, height-1)
                x = random.randint(0, width-1)
                radius = random.randint(1, 2)
                intensity = random.uniform(0.008, 0.015)  # 降低强度
                
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < height and 0 <= nx < width:
                            dist = math.sqrt(dx**2 + dy**2)
                            if dist <= radius:
                                pattern[ny, nx] += intensity * (1 - dist/radius)
        
        else:  # random
            # 更轻微的随机缺陷
            noise = torch.randn(height, width, 3) * 0.003
            pattern += torch.clamp(noise, 0, 0.01)
        
        return pattern

    def apply_sensor_characteristics(self, img_tensor, sensor_profile, iso, noise_level, 
                                    defect_pattern, defect_strength, noise_strength, device,
                                    enable_sensor_noise, enable_sensor_defects):
        """应用传感器特性和缺陷 (减轻效果)"""
        # 基础传感器噪声
        if enable_sensor_noise:
            if sensor_profile in TikTokAntiDetectUtils.sensor_params:
                img_tensor = TikTokAntiDetectUtils.apply_sensor_noise(
                    img_tensor, 
                    noise_strength * (iso/800)**0.5 * 0.8,  # 降低噪声强度
                    sensor_profile, 
                    iso, 
                    device,
                    in_latent_space=False
                )
            else:
                img_tensor = TikTokAntiDetectUtils.apply_sensor_noise(
                    img_tensor, 
                    noise_strength * (iso/800)**0.5 * 0.8,  # 降低噪声强度
                    "dslr", 
                    iso, 
                    device,
                    in_latent_space=False
                )
        
        # 添加特定缺陷模式
        if enable_sensor_defects:
            h, w = img_tensor.shape[1:3]
            defect_resized = TF.resize(defect_pattern.permute(2, 0, 1), (h, w)).permute(1, 2, 0).to(device)
            img_tensor = img_tensor + defect_strength * 0.8 * defect_resized  # 降低缺陷强度
        
        return img_tensor

    def apply_quantum_protection(self, img_tensor, strength, frame_id, device):
        """量子保护 (减轻变暗效果)"""
        protected = img_tensor.clone()
        h, w = img_tensor.shape[1:3]
        
        # 使用更明亮的量子签名
        signature = TikTokAntiDetectUtils.generate_quantum_signature(frame_id, h, w)
        
        # 应用到每个通道
        for c in range(3):
            channel = img_tensor[0, :, :, c].cpu().numpy()
            
            # FFT变换
            fft_channel = np.fft.fft2(channel)
            
            # 应用量子签名 (提升低频亮度)
            protected_fft = fft_channel * (signature * 1.2)  # 增强亮度
            
            # 逆FFT
            quantum_layer = np.abs(np.fft.ifft2(protected_fft))
            
            # 混合结果 (减轻混合强度)
            quantum_tensor = torch.from_numpy(quantum_layer).float().to(device)
            protected[0, :, :, c] = protected[0, :, :, c] * (1 - strength) + quantum_tensor * strength * 0.7
        
        return protected

    def add_lens_flare(self, img_tensor, strength, device):
        """添加镜头炫光效果 (减轻亮度)"""
        if strength < 0.01:
            return img_tensor
            
        batch, height, width, _ = img_tensor.shape
        
        # 随机光源位置
        light_x = random.uniform(0.1, 0.9)
        light_y = random.uniform(0.1, 0.9)
        
        # 创建炫光效果
        flare = torch.zeros((height, width), device=device)
        
        # 主光斑 (减小尺寸)
        main_radius = min(height, width) * 0.03  # 减小半径
        y, x = torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device), 
            indexing='ij'
        )
        dist_to_light = torch.sqrt((x/width - light_x)**2 + (y/height - light_y)**2)
        main_flare = torch.exp(-dist_to_light**2 / (2*(main_radius/3)**2)) * 0.3  # 降低强度
        
        # 次光斑 (减少数量和强度)
        num_secondary = random.randint(1, 3)  # 减少数量
        for i in range(num_secondary):
            offset_x = random.uniform(-0.2, 0.2)  # 减小偏移范围
            offset_y = random.uniform(-0.2, 0.2)
            scale = random.uniform(0.15, 0.35)  # 减小尺寸
            intensity = random.uniform(0.1, 0.25)  # 降低强度
            
            dist = torch.sqrt((x/width - (light_x + offset_x))**2 + 
                             (y/height - (light_y + offset_y))**2)
            secondary = torch.exp(-dist**2 / (2*(main_radius*scale/3)**2)) * intensity
            flare += secondary
        
        # 组合炫光效果
        flare = (main_flare + flare) * strength * 0.5  # 大幅降低整体强度
        
        # 应用炫光 (使用更柔和的混合)
        flare_rgb = flare.unsqueeze(-1).repeat(1, 1, 3)
        return img_tensor + flare_rgb * 0.5  # 减轻混合强度

    def add_chromatic_aberration(self, img_tensor, strength, device):
        """添加色差效果 (减轻效果)"""
        if strength < 0.01:
            return img_tensor
            
        batch, height, width, _ = img_tensor.shape
        
        # 随机色差方向
        offset_angle = random.uniform(0, 2*math.pi)
        offset_scale = strength * 0.02  # 减小位移比例
        offset_r = offset_scale * 0.8
        offset_g = offset_scale * 1.0
        offset_b = offset_scale * 1.2
        
        # 中心点位移最小，边缘最大
        center_x, center_y = width/2, height/2
        y, x = torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device), 
            indexing='ij'
        )
        dist_to_center = torch.sqrt(((x - center_x)/width)**2 + ((y - center_y)/height)**2)
        
        # 应用位移 (减小位移量)
        dx = dist_to_center * torch.cos(torch.tensor(offset_angle)) * offset_b * 0.7
        dy = dist_to_center * torch.sin(torch.tensor(offset_angle)) * offset_b * 0.7
        
        # 分离通道
        r_channel = img_tensor[..., 0:1]
        g_channel = img_tensor[..., 1:2]
        b_channel = img_tensor[..., 2:3]
        
        # 应用不同位移
        r_offset = self.warp_channel(r_channel, dx * offset_r, dy * offset_r, device)
        g_offset = self.warp_channel(g_channel, dx * offset_g, dy * offset_g, device)
        b_offset = self.warp_channel(b_channel, dx * offset_b, dy * offset_b, device)
        
        # 重新组合通道
        return torch.cat([r_offset, g_offset, b_offset], dim=-1)

    def warp_channel(self, channel, dx, dy, device):
        """扭曲单通道图像"""
        batch, height, width, _ = channel.shape
        
        y, x = torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device), 
            indexing='ij'
        )
        x_norm = x.float() / (width - 1) * 2 - 1
        y_norm = y.float() / (height - 1) * 2 - 1
        
        x_new = torch.clamp(x_norm + dx, -1, 1)
        y_new = torch.clamp(y_norm + dy, -1, 1)
        
        grid = torch.stack((x_new, y_new), dim=-1).unsqueeze(0)
        return torch.nn.functional.grid_sample(
            channel.permute(0, 3, 1, 2), grid, 
            align_corners=True, mode='bilinear', padding_mode='reflection'
        ).permute(0, 2, 3, 1)

    def compress_dynamic_range(self, img_tensor, strength, device_dr, device):
        """压缩动态范围 (减轻效果)"""
        if strength < 0.01:
            return img_tensor
            
        # 目标动态范围
        target_dr = 11.0 + (device_dr - 11.0) * (1 - strength)  # 提高基础值
        
        # 应用更温和的S型曲线
        compressed = self.apply_s_curve(img_tensor, target_dr)
        
        return torch.clamp(compressed, 0.02, 0.98)  # 减小剪切范围

    def apply_s_curve(self, img, target_dr):
        """应用更温和的S型曲线"""
        # 更平缓的曲线
        k = 0.6 + (target_dr / 20.0) * 0.4  # 减小斜率
        
        # 更温和的变换
        return 1 / (1 + torch.exp(-k * (img * 6 - 3)))

    def apply_h265_compression(self, img_tensor, strength, device):
        """模拟H.265压缩伪影 (减轻效果)"""
        if strength < 0.01:
            return img_tensor
            
        # 转换为numpy处理
        img_np = img_tensor.detach().cpu().numpy()[0]
        img_uint8 = (img_np * 255).astype(np.uint8)
        
        # 模拟H.265编码 (提高质量)
        quality = max(50, int(95 - strength*45))  # 减小质量下降幅度
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), encode_param)
        compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB) / 255.0
        
        # 添加更轻微的块效应
        if strength > 0.2:
            block_size = 8
            grid_h = compressed.shape[0] // block_size
            grid_w = compressed.shape[1] // block_size
            
            for i in range(grid_h):
                for j in range(grid_w):
                    if random.random() < strength * 0.25:  # 降低块效应概率
                        y_start = i * block_size
                        x_start = j * block_size
                        block = compressed[y_start:y_start+block_size, x_start:x_start+block_size]
                        
                        # 减小边界对比度增强
                        block[0, :] *= 1.02
                        block[-1, :] *= 1.02
                        block[:, 0] *= 1.02
                        block[:, -1] *= 1.02
                        
                        # 内部平滑
                        if block_size > 4:
                            inner = block[1:-1, 1:-1]
                            inner[:] = gaussian_filter(inner, sigma=0.3)  # 减小平滑强度
        
        # 添加更轻微的色度子采样伪影
        if strength > 0.3:
            yuv = cv2.cvtColor((compressed * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
            u = cv2.resize(yuv[:,:,1], (compressed.shape[1]//2, compressed.shape[0]//2), interpolation=cv2.INTER_AREA)
            v = cv2.resize(yuv[:,:,2], (compressed.shape[1]//2, compressed.shape[0]//2), interpolation=cv2.INTER_AREA)
            u = cv2.resize(u, (compressed.shape[1], compressed.shape[0]), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (compressed.shape[1], compressed.shape[0]), interpolation=cv2.INTER_LINEAR)
            yuv[:,:,1] = u
            yuv[:,:,2] = v
            compressed = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB) / 255.0
        
        # 混合原始图像以保留细节 (增加原始图像比例)
        return torch.from_numpy(compressed).float().unsqueeze(0).to(device) * strength * 0.7 + img_tensor * (1 - strength * 0.7)

    def apply_color_response(self, img_tensor, color_response, strength, device):
        """应用设备特定的色彩响应 (减轻效果)"""
        if strength < 0.01:
            return img_tensor
            
        response = torch.tensor(color_response, device=device).view(1, 1, 1, 3)
        # 更柔和的混合方式，避免过度增强
        return img_tensor * (1 - strength) + (img_tensor * response) * strength

    def inject_exif_metadata(self, images, device_model, iso):
        """注入EXIF元数据"""
        batch_size, height, width, _ = images.shape
        
        # 创建元数据水印
        metadata = f"{device_model}|ISO:{iso}|{hashlib.md5(device_model.encode()).hexdigest()[:8]}"
        binary_meta = ''.join(format(ord(c), '08b') for c in metadata)
        
        # 应用水印到所有图像
        for i in range(batch_size):
            img = images[i].cpu().numpy()
            
            for bit_idx, bit in enumerate(binary_meta):
                if bit_idx >= height * width * 3:
                    break
                
                h = bit_idx // (width * 3)
                w = (bit_idx % (width * 3)) // 3
                c = bit_idx % 3
                
                if h < height and w < width:
                    pixel_val = img[h, w, c]
                    int_val = int(pixel_val * 255)
                    int_val &= 0xFE
                    if bit == '1':
                        int_val |= 0x01
                    img[h, w, c] = int_val / 255.0
            
            images[i] = torch.from_numpy(img)
        
        return images

# # 节点映射
# NODE_CLASS_MAPPINGS = {
#     "ZhuqueAntiDetectEnhancer": ZhuqueAntiDetectEnhancer,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "ZhuqueAntiDetectEnhancer": "朱雀AI检测对抗增强器",
# }