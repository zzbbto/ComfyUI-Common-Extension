import torch
import gc
import sys
import time
import ctypes
from comfy import model_management
from typing import Any, Optional, Tuple
from server import PromptServer
import weakref
from types import ModuleType, FunctionType
from collections import defaultdict

# ========================== 统一清理接口 ==========================

def unload_models(use_comfy_method: bool = True, send_signal: bool = True) -> None:
    """卸载所有模型资源"""
    print("♻️ 开始卸载模型...")
    
    # 1. 使用ComfyUI的卸载方法
    if use_comfy_method:
        if hasattr(model_management, 'unload_all_models'):
            model_management.unload_all_models()
        if hasattr(model_management, 'soft_empty_cache'):
            model_management.soft_empty_cache()
    
    # 2. 额外的手动清理
    # _clean_model_references()
    
    # 3. 发送清理信号
    if send_signal:
        try:
            if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                PromptServer.instance.send_sync("memory_cleanup", {
                    "type": "cleanup_request",
                    "data": {
                        "unload_models": True,
                        "free_memory": True,
                        "force": True
                    }
                })
                print("已发送增强内存清理信号")
        except Exception as e:
            print(f"⚠️ 发送内存清理信号失败: {str(e)}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    print("✅ 模型卸载完成")

def clear_memory(intensity: int = 3) -> Tuple[float, float]:
    """清理显存资源并返回清理前后的显存使用量"""
    print(f"♻️ 开始显存清理(强度 {intensity})...")
    
    # 记录清理前状态
    pre_vram = get_system_vram_usage()
    
    # 1. 基本清理
    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()
    
    # 2. 中级清理
    if intensity >= 2:
        for _ in range(3):
            gc.collect()
            gc.collect(generation=2)  # 清理老一代对象
    
    # 3. 高级清理
    if intensity >= 3:
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        
        # 清理CUDA IPC资源
        try:
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"⚠️ IPC收集失败: {str(e)}")
        
        # # 清理NVIDIA驱动缓存
        # try:
        #     _clean_nvidia_driver_cache()
        # except Exception as e:
        #     print(f"⚠️ NVIDIA驱动缓存清理失败: {str(e)}")
    
    # # 4. 专家级清理
    # if intensity >= 4:
    #     try:
    #         # 同步所有设备
    #         for i in range(torch.cuda.device_count()):
    #             torch.cuda.synchronize(i)
            
    #         # 强制释放PyTorch保留的缓存
    #         if hasattr(torch.cuda, 'memory_reserved'):
    #             torch.cuda.memory_reserved(0)
    #             torch.cuda.memory_reserved(0)  # 双重调用确保释放
            
    #         # 使用低级API释放内存
    #         _release_cuda_caches()
            
    #         # 强制断开Python对象引用
    #         _break_object_references()
    #     except Exception as e:
    #         print(f"⚠️ 专家级清理失败: {str(e)}")
    
    # # 5. 终极清理（针对顽固内存）
    # if intensity >= 5:
    #     try:
    #         # 执行终极断开引用操作
    #         print("🔨 执行终极断开引用操作...")
    #         _break_all_cuda_references()
            
    #         # 强制重置CUDA上下文
    #         _reset_cuda_context_aggressive()
    #         print("♻️ 已执行终极CUDA上下文重置")
    #     except Exception as e:
    #         print(f"⚠️ 终极清理失败: {str(e)}")
    
    # 记录清理后状态
    post_vram = get_system_vram_usage()
    freed = pre_vram - post_vram
    
    print(f"✅ 显存清理完成(强度 {intensity}), 释放 {freed:.2f}MB")
    return pre_vram, post_vram

def perform_full_cleanup(unload: bool = True, intensity: int = 4) -> Tuple[float, float]:
    """执行完整的清理流程：卸载模型 + 清理显存"""
    pre_ram = get_system_vram_usage()
    
    if unload:
        unload_models()
    
    pre_vram, post_vram = clear_memory(intensity)
    
    post_ram = get_system_vram_usage()
    return pre_ram, post_ram, pre_vram, post_vram

# def ultimate_cleanup(warn: bool = True, reinit: bool = True) -> Optional[float]:
#     """
#     终极显存清理：完全释放CUDA上下文和所有显存
#     返回：清理后剩余显存(MB)或None(如果失败)
#     """
#     if warn:
#         print("⚠️"*50)
#         print("警告: 即将执行终极显存清理!")
#         print("此操作将完全释放CUDA上下文和所有显存")
#         print("可能导致当前工作流中的模型和状态丢失")
#         print("仅在严重内存泄漏或需要完全重置时使用")
#         print("⚠️"*50)
#         time.sleep(3)  # 给用户时间阅读警告
    
#     try:
#         # 1. 记录初始显存状态
#         initial_vram = get_system_vram_usage()
        
#         # 2. 卸载所有模型
#         unload_models()
        
#         # 3. 执行最高强度清理
#         clear_memory(intensity=5)
        
#         # 4. 使用PyTorch内部重置
#         if hasattr(torch.cuda, '_reset'):
#             torch.cuda._reset()
#             print("✅ PyTorch CUDA完全重置完成")
#         else:
#             print("⚠️ PyTorch版本不支持完全重置")
        
#         # 5. 系统级CUDA驱动清理
#         _reset_cuda_driver()
        
#         # 6. 最终清理
#         for _ in range(3):
#             gc.collect()
#             torch.cuda.empty_cache()
        
#         # 7. 重新初始化CUDA（可选）
#         if reinit:
#             torch.cuda.init()
#             print("✅ CUDA重新初始化完成")
        
#         # 8. 报告结果
#         final_vram = get_system_vram_usage()
#         freed = initial_vram - final_vram
#         print(f"♻️ 显存释放结果: 初始 {initial_vram:.1f}MB -> 最终 {final_vram:.1f}MB | 释放 {freed:.1f}MB")
        
#         return final_vram
#     except Exception as e:
#         print(f"⚠️ 终极清理失败: {str(e)}")
#         traceback.print_exc()
#         return None

# ========================== 辅助函数 ==========================

# def _clean_nvidia_driver_cache() -> None:
#     """清理NVIDIA驱动缓存"""
#     # Windows
#     if sys.platform == 'win32':
#         cache_path = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'NVIDIA', 'DXCache')
#         if os.path.exists(cache_path):
#             for file in os.listdir(cache_path):
#                 try:
#                     file_path = os.path.join(cache_path, file)
#                     # 只删除缓存文件，不删除目录
#                     if os.path.isfile(file_path) and file.endswith('.bin'):
#                         os.remove(file_path)
#                 except Exception as e:
#                     print(f"⚠️ 删除缓存文件失败: {file_path}, {str(e)}")
#     # Linux
#     elif sys.platform == 'linux':
#         cache_path = '/var/cache/nvidia'
#         if os.path.exists(cache_path):
#             for file in os.listdir(cache_path):
#                 try:
#                     file_path = os.path.join(cache_path, file)
#                     if os.path.isfile(file_path) and file.endswith('.bin'):
#                         os.remove(file_path)
#                 except Exception as e:
#                     print(f"⚠️ 删除缓存文件失败: {file_path}, {str(e)}")

def _release_cuda_caches() -> None:
    """释放CUDA内部缓存"""
    # 清理cuBLAS工作空间
    if hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
        torch._C._cuda_clearCublasWorkspaces()
    
    # 清理cuDNN缓存
    if hasattr(torch.backends.cudnn, 'clear'):
        torch.backends.cudnn.clear()
    
    # 使用更高级的方法清理缓存分配器
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

def _break_object_references() -> None:
    """安全高效地断开CUDA张量引用 (优化版)"""
    print("♻️ 正在断开Python对象引用...")
    broken_refs = 0
    
    # 第一轮：识别高危对象
    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_tensors.append(weakref.ref(obj))
        except Exception:
            continue
    
    # 第二轮：针对性处理容器
    for obj in gc.get_objects():
        # 跳过不可修改对象
        if isinstance(obj, (ModuleType, type, FunctionType, weakref.ref)):
            continue
            
        # 列表处理
        if isinstance(obj, list):
            for i in range(len(obj)):
                try:
                    item = obj[i]
                    # 检查是否是CUDA张量
                    if torch.is_tensor(item) and item.is_cuda:
                        obj[i] = None
                        broken_refs += 1
                except Exception:
                    continue
        
        # 字典处理
        elif isinstance(obj, dict):
            # 先收集需要删除的键
            keys_to_remove = []
            for key, val in obj.items():
                try:
                    # 检查是否是CUDA张量
                    if torch.is_tensor(val) and val.is_cuda:
                        keys_to_remove.append(key)
                except Exception:
                    continue
            
            # 安全删除键值对
            for key in keys_to_remove:
                try:
                    del obj[key]
                    broken_refs += 1
                except Exception:
                    pass
    
    # 自定义对象处理 (安全模式)
    for obj in gc.get_objects():
        try:
            # 跳过模块对象，避免触发__getattr__错误
            if isinstance(obj, ModuleType):
                continue
                
            # 跳过transformers相关对象
            module_name = getattr(obj, '__module__', '')
            if 'transformers' in module_name:
                continue
                
            if hasattr(obj, '__dict__') and not isinstance(obj, type):
                # 仅处理简单属性，避免触发复杂逻辑
                for attr in list(vars(obj).copy()):  # 使用副本避免修改问题
                    # 跳过双下划线属性
                    if attr.startswith('__') and attr.endswith('__'):
                        continue
                        
                    try:
                        # 安全获取属性值
                        val = getattr(obj, attr)
                        if torch.is_tensor(val) and val.is_cuda:
                            setattr(obj, attr, None)
                            broken_refs += 1
                    except Exception:
                        # 忽略所有属性访问错误
                        continue
        except Exception:
            # 忽略对象访问错误
            continue

    print(f"已断开 {broken_refs} 个张量引用")
    
    # 分阶段回收
    for _ in range(2):
        gc.collect()
        torch.cuda.empty_cache()

def _break_all_cuda_references() -> None:
    """断开所有可能的CUDA张量引用"""
    print("🔨 正在断开所有可能的CUDA张量引用...")
    
    # 1. 查找所有持有CUDA张量的对象
    cuda_holders = []
    for obj in gc.get_objects():
        try:
            # 跳过模块对象，避免触发__getattr__错误
            if isinstance(obj, ModuleType):
                continue
                
            # 跳过transformers相关对象
            module_name = getattr(obj, '__module__', '')
            if 'transformers' in module_name:
                continue
                
            if isinstance(obj, list):
                for item in obj:
                    if torch.is_tensor(item) and item.is_cuda:
                        cuda_holders.append(obj)
                        break
            elif isinstance(obj, dict):
                for val in obj.values():
                    if torch.is_tensor(val) and val.is_cuda:
                        cuda_holders.append(obj)
                        break
            elif hasattr(obj, '__dict__'):
                # 仅处理简单属性，避免触发复杂逻辑
                for attr in list(vars(obj).copy()):  # 使用副本避免修改问题
                    # 跳过双下划线属性
                    if attr.startswith('__') and attr.endswith('__'):
                        continue
                        
                    try:
                        val = getattr(obj, attr)
                        if torch.is_tensor(val) and val.is_cuda:
                            cuda_holders.append(obj)
                            break
                    except Exception:
                        # 忽略所有属性访问错误
                        continue
        except Exception:
            continue
    
    print(f"发现 {len(cuda_holders)} 个持有CUDA张量的对象")
    
    # 2. 强制清除这些对象
    for holder in cuda_holders:
        try:
            if isinstance(holder, list):
                # 安全清空列表
                holder.clear()
            elif isinstance(holder, dict):
                # 安全清空字典
                holder.clear()
            elif hasattr(holder, '__dict__'):
                # 清除对象属性
                for attr in list(vars(holder).copy()):  # 使用副本避免修改问题
                    # 跳过双下划线属性
                    if attr.startswith('__') and attr.endswith('__'):
                        continue
                        
                    try:
                        val = getattr(holder, attr)
                        if torch.is_tensor(val) and val.is_cuda:
                            setattr(holder, attr, None)
                    except Exception:
                        pass
        except Exception as e:
            print(f"⚠️ 清理对象失败: {str(e)}")
    
    # 3. 强制垃圾回收
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"已断开 {len(cuda_holders)} 个持有CUDA张量的对象引用")

def _reset_cuda_context_aggressive() -> None:
    """终极CUDA上下文重置方案（安全版）"""
    try:
        # 确保pynvml可用
        pynvml = install_pynvml()
        if not pynvml:
            print("⚠️ pynvml不可用，跳过NVML操作")
            # 执行基本CUDA清理
            _fallback_cuda_cleanup()
            return
            
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError_LibraryNotFound:
            print("⚠️ NVML库未找到，跳过NVML操作")
            _fallback_cuda_cleanup()
            return
        except pynvml.NVMLError_InsufficientPermissions:
            print("⚠️ 权限不足，无法访问NVML，跳过NVML操作")
            _fallback_cuda_cleanup()
            return
        except Exception as e:
            print(f"⚠️ NVML初始化失败: {str(e)}")
            _fallback_cuda_cleanup()
            return
        
        # 2. 获取设备句柄
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            print(f"♻️ 正在重置 {device_count} 个设备的CUDA上下文...")
        except pynvml.NVMLError_NotSupported:
            print("⚠️ 设备不支持NVML操作")
            device_count = 0
        
        # 3. 重置每个设备的上下文 - 仅执行安全操作
        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 3.3 重置内存锁 (通常不需要特权)
                # 修复：跳过CPU亲和性操作，避免参数错误
                # 改为使用更安全的设备状态重置
                try:
                    # 重置计算模式（如果支持）
                    current_mode = pynvml.nvmlDeviceGetComputeMode(handle)
                    pynvml.nvmlDeviceSetComputeMode(handle, pynvml.NVML_COMPUTEMODE_DEFAULT)
                    print(f"✅ 设备 {i} 计算模式已重置")
                except pynvml.NVMLError_NotSupported:
                    print(f"⚠️ 设备 {i} 不支持计算模式操作")
                except Exception as e:
                    print(f"⚠️ 设备 {i} 计算模式操作失败: {str(e)}")
            except pynvml.NVMLError_NoPermission:
                print(f"⚠️ 设备 {i} 权限不足，跳过操作")
            except pynvml.NVMLError_NotSupported:
                print(f"⚠️ 设备 {i} 不支持此操作")
            except Exception as e:
                print(f"⚠️ 设备 {i} 操作失败: {str(e)}")
        
        # 4. 强制重置PyTorch的CUDA状态
        _fallback_cuda_cleanup()
        
        print("✅ 终极CUDA上下文重置完成")
    except Exception as e:
        print(f"⚠️ 终极CUDA上下文重置失败: {str(e)}")
        # 回退到基本清理
        _fallback_cuda_cleanup()
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def _fallback_cuda_cleanup() -> None:
    """CUDA清理回退方案"""
    print("♻️ 执行回退CUDA清理...")
    try:
        # 修复：正确检测PyTorch版本
        if hasattr(torch.cuda, '_reset'):
            # 使用PyTorch内置的完整重置
            torch.cuda._reset()
            print("✅ 已执行PyTorch CUDA完全重置")
        else:
            # 对于旧版本PyTorch的兼容方案
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats(i)
        
        # 多次强制释放所有CUDA IPC资源
        for _ in range(3):
            try:
                torch.cuda.ipc_collect()
                time.sleep(0.1)
            except Exception:
                pass
        
        # 系统级内存清理
        if sys.platform == 'win32':
            # Windows: 强制释放工作集内存
            try:
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.c_ulong(-1))
                print("✅ Windows工作集内存已清理")
            except Exception as e:
                print(f"⚠️ Windows工作集清理失败: {str(e)}")
        elif sys.platform == 'linux':
            # Linux: 使用malloc_trim
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
                print("✅ Linux malloc_trim已执行")
            except Exception as e:
                print(f"⚠️ Linux malloc_trim失败: {str(e)}")
        
        # 额外清理步骤：释放Python内部缓存
        try:
            sys._clear_type_cache()
            gc.collect()
        except Exception:
            pass
            
    except Exception as e:
        print(f"⚠️ 回退CUDA清理失败: {str(e)}")

def _clean_model_references() -> None:
    """高效清理残留的模型引用，避免递归操作"""
    print("🧹 快速清理模型引用...")
    models_to_clean = []
    optimizers_to_clean = []
    
    objects = list(gc.get_objects())  # 先获取所有对象列表
    total_objects = len(objects)
    
    # 第一次快速遍历：只收集需要清理的对象
    for i, obj in enumerate(objects):
        if i % 1000000 == 0 and total_objects > 1000000:  # 每100000个对象报告一次进度
            print(f"  扫描进度: {i}/{total_objects} ({i/total_objects*100:.1f}%)")
        
        try:
            # 跳过模块对象，避免触发__getattr__错误
            if isinstance(obj, ModuleType):
                continue
                
            # 跳过transformers相关对象
            module_name = getattr(obj, '__module__', '')
            if 'transformers' in module_name:
                continue
                
            if isinstance(obj, torch.nn.Module):
                models_to_clean.append(weakref.ref(obj))
            elif isinstance(obj, torch.optim.Optimizer):
                optimizers_to_clean.append(weakref.ref(obj))
        except Exception:
            continue
    
    print(f"发现 {len(models_to_clean)}个模型和 {len(optimizers_to_clean)}个优化器需要清理")
    
    # 清理模型
    for i, model_ref in enumerate(models_to_clean):
        model = model_ref()
        if model is None:
            continue
            
        try:
            # 每1000个模型报告一次进度
            if i % 1000 == 0:
                print(f"  清理模型: {i}/{len(models_to_clean)}")
            
            # 安全清除模型参数
            for name, param in list(model.named_parameters(recurse=False)):
                if param is not None:
                    param.data = torch.empty(0, device='cpu')
                    if param.grad is not None:
                        param.grad.data = torch.empty(0, device='cpu')
            
            # 清除子模块引用
            for name, child in list(model.named_children()):
                setattr(model, name, None)
                
            # 清除缓存
            if hasattr(model, '_parameters'):
                model._parameters.clear()
            if hasattr(model, '_buffers'):
                model._buffers.clear()
            if hasattr(model, '_modules'):
                model._modules.clear()
                
        except Exception as e:
            # print(f"⚠️ 清理模型失败: {str(e)}")
            pass
    
    # 清理优化器
    for i, opt_ref in enumerate(optimizers_to_clean):
        optimizer = opt_ref()
        if optimizer is None:
            continue
            
        try:
            # 清除优化器状态
            optimizer.state = defaultdict(dict)
            # 清除参数组
            optimizer.param_groups = []
        except Exception:
            pass
    
    # 强制执行垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ 清理完成: {len(models_to_clean)}个模型, {len(optimizers_to_clean)}个优化器")
   

def check_psutil() -> bool:
    try:
        import psutil
        return True
    except ImportError:
        print("⚠️ psutil 未安装，无法获取系统内存信息")
        return False
    
def install_pynvml():
    try:
        import pynvml
        return pynvml
    except ImportError:
        print("⚠️ pynvml未安装，跳过NVML操作")
        return None
    except Exception as e:
        print(f"⚠️ 导入pynvml失败: {str(e)}")
        return None

def get_system_vram_usage() -> float:
    """获取准确的系统总显存使用量（包含驱动层）"""
    total_used_mb = 0
    
    # 方案1: 使用pynvml获取精确值（首选）
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_used_mb += info.used / (1024 * 1024)
        pynvml.nvmlShutdown()
        return total_used_mb
    except ImportError:
        pass
    
    # 方案2: 解析nvidia-smi输出（跨平台）
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # 处理多GPU情况
            usages = result.stdout.strip().split('\n')
            for usage in usages:
                if usage.isdigit():
                    total_used_mb += float(usage)
            return total_used_mb
    except:
        pass
    
    # 方案3: 回退到PyTorch统计（不准确但可用）
    return torch.cuda.memory_allocated() / (1024 ** 2) + 500  # 加上估计的上下文占用

# 新增函数：获取显卡总显存容量
def get_total_gpu_memory() -> float:
    """获取显卡总显存容量（MB）"""
    try:
        # 首选方法：使用pynvml
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认第一个GPU
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mb = info.total / (1024 ** 2)
        pynvml.nvmlShutdown()
        return total_mb
    except ImportError:
        pass
    except Exception:
        pass
    
    # 备选方法：使用nvidia-smi命令行
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    
    # 最后备选：使用PyTorch的近似值
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    
    # 默认值（常见显卡容量）
    return 8192.0  # 8GB
    
def _reset_cuda_driver():
    """直接重置CUDA驱动状态（系统级操作）"""
    try:
        if sys.platform == 'win32':
            # Windows: 卸载并重新加载nvcuda.dll
            _reset_cuda_windows()
        elif sys.platform == 'linux':
            # Linux: 使用dlclose强制卸载CUDA驱动
            _reset_cuda_linux()
        else:
            print("⚠️ 不支持的操作系统")
    except Exception as e:
        print(f"⚠️ CUDA驱动重置失败: {str(e)}")

def _reset_cuda_windows():
    """Windows下的CUDA驱动重置"""
    try:
        # 1. 获取当前进程句柄
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        current_process = kernel32.GetCurrentProcess()
        
        # 2. 尝试卸载nvcuda.dll
        nvcuda = ctypes.WinDLL('nvcuda')
        handle = nvcuda._handle
        
        # 3. 使用FreeLibrary卸载驱动
        if kernel32.FreeLibrary(handle):
            print("✅ nvcuda.dll已卸载")
        else:
            print("⚠️ 无法卸载nvcuda.dll")
        
        # 4. 重新加载CUDA驱动
        torch.cuda.init()
        print("✅ CUDA驱动重新加载完成")
    except Exception as e:
        print(f"⚠️ Windows CUDA重置失败: {str(e)}")
        # 回退到基本初始化
        torch.cuda.init()

def _reset_cuda_linux():
    """Linux下的CUDA驱动重置"""
    try:
        # 1. 使用dlclose卸载CUDA库
        libdl = ctypes.CDLL("libdl.so.2")
        libcuda = ctypes.CDLL("libcuda.so")
        
        # 2. 获取库句柄
        handle = libcuda._handle
        
        # 3. 卸载库
        if libdl.dlclose(handle) == 0:
            print("✅ libcuda.so已卸载")
        else:
            print("⚠️ 无法卸载libcuda.so")
        
        # 4. 重新加载CUDA驱动
        torch.cuda.init()
        print("✅ CUDA驱动重新加载完成")
    except Exception as e:
        print(f"⚠️ Linux CUDA重置失败: {str(e)}")
        # 回退到基本初始化
        torch.cuda.init()