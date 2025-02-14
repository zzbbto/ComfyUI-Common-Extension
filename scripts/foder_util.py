import os
import torch
from PIL import Image, ImageOps
import numpy as np

class LoadFolderFromDirList:
    # 描述输入参数的类型和约束（可选，取决于您的使用场景）
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_file_name1": ("STRING", {"default": None}),
                "include_file_name2": ("STRING", {"default": None}),
                "include_file_name3": ("STRING", {"default": None}),       
                "all_satisfy": ("BOOLEAN", {"default": True}),    
                "ignore_file_name1": ("STRING", {"default": None}),                                              
            }
        }

    # 描述返回类型和名称（可选，取决于您的使用场景）
    RETURN_TYPES = ("STRING", "STRING","INT")
    RETURN_NAMES = ("文件夹全路径列表", "文件夹名列表","数量")
    OUTPUT_IS_LIST = (True, True, False)

    FUNCTION = "load_folders"
    CATEGORY = "CommonExtension/folder"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def load_folders(self, directory: str, image_load_cap: int = 0, start_index: int = 0,
                     include_file_name1: str = None, include_file_name2: str = None,
                     include_file_name3: str = None, all_satisfy: bool = True,
                     ignore_file_name1: str = None):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
        folder_paths = []
        folder_names = []
        image_count = 0
        limit_images = image_load_cap > 0
        current_index = 0
        
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.is_dir():
                        if limit_images and image_count >= image_load_cap:
                            break                        
                        if(not is_none_or_empty(ignore_file_name1) and ignore_file_name1 in entry.name):
                            continue

                        file_name_with_ext = entry.name
                        # 处理文件名包含逻辑
                        match = False
                        if all_satisfy:
                            # 如果all_satisfy为True，则所有非空的include_file_name都必须匹配
                            match = (is_none_or_empty(include_file_name1) or include_file_name1 in file_name_with_ext) and \
                                   (is_none_or_empty(include_file_name2) or include_file_name2 in file_name_with_ext) and \
                                   (is_none_or_empty(include_file_name3) or include_file_name3 in file_name_with_ext)
                        else:
                            # 如果all_satisfy为False，则至少一个非空的include_file_name需要匹配
                            match = (not is_none_or_empty(include_file_name1) and include_file_name1 in file_name_with_ext) or \
                                   (not is_none_or_empty(include_file_name2) and include_file_name2 in file_name_with_ext) or \
                                   (not is_none_or_empty(include_file_name3) and include_file_name3 in file_name_with_ext)

                        if match:
                            current_index += 1
                            if start_index<current_index:
                                folder_paths.append(entry.path)
                                folder_names.append(file_name_with_ext)
                                image_count += 1  # 注意：这里假设每个文件夹至少包含一个图像，可能需要调整逻辑
        except Exception as e:
            # 捕获并处理其他潜在的异常（比如权限问题）
            # 在生产代码中，应该有更具体的错误处理逻辑
            raise e
        
        if not folder_paths:
            raise FileNotFoundError(f"No folders found in directory '{directory}' starting from index {start_index}.")
        
        # 返回一个包含文件夹路径、文件夹名和加载的图像数量的元组，以及一个表示是否加载了图像的布尔值
        return folder_paths, folder_names, image_count

def check_have_file(directory: str, file_name: str) -> bool:
    print(f"check_have_file {directory}，{file_name} ")
    if not os.path.isdir(directory):
        return False
    try:
        with os.scandir(directory) as it:
            for entry in it:
                # print(f"target_name = {entry.name} ")
                if entry.is_file() and file_name in entry.name:
                    # print(f"check_have_file:file_name = {file_name},target_name = {entry.name}, True ")
                    return True
    except Exception as e:
        # 捕获并处理其他潜在的异常（比如权限问题），这里只是打印错误日志
        print(f"Error occurred while checking directory: {e}")
        return False
    return False

def is_none_or_empty(value):
    if value is None or value == "":
        return True
    else:
        return False


class LoadFilesFromDirList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_file_name1": ("STRING", {"default": None}),
                "include_file_name2": ("STRING", {"default": None}),
                "include_file_name3": ("STRING", {"default": None}),       
                "all_satisfy": ("BOOLEAN", {"default": True}),     
                "check_desdir_havefile": ("STRING", {"default": None}),         
                "file_add_name": ("STRING", {"default": None}),    
                "ignore_file_name1": ("STRING", {"default": None}),                                    
            }
        }
    
    # 更新返回类型和名称以匹配实际返回的值
    RETURN_TYPES = ("IMAGE","STRING", "STRING", "STRING", "STRING", "STRING","INT")
    RETURN_NAMES = ("图像","上一级目录名","文件全路径(带后缀名)", "文件全路径(不带后缀名)", "文件名(带后缀名)", "文件名(不带后缀名)", "数量")
    OUTPUT_IS_LIST = (True, True, True,True,True,True,False)

    FUNCTION = "load_files"
    CATEGORY = "CommonExtension/file"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def load_files(self, directory: str, image_load_cap: int = 0, start_index: int = 0,
                    include_file_name1: str = None, include_file_name2: str = None,include_file_name3: str = None, all_satisfy: bool = True,
                    check_desdir_havefile: str = None,file_add_name: str = None,
                    ignore_file_name1: str = None):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
        # 初始化列表以存储文件路径和名称
        images = []
        parent_folder_names = []
        full_paths_with_ext = []
        full_paths_without_ext = []
        file_names_with_ext = []
        file_names_without_ext = []

        # 使用计数器来跳过文件直到start_index达到
        current_index = 0
        image_count = 0
        limit_images = image_load_cap > 0

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.is_file(): 
                        if limit_images and image_count >= image_load_cap:
                            break
                        if(not is_none_or_empty(ignore_file_name1) and ignore_file_name1 in entry.name):
                            continue      

                        file_name_with_ext = entry.name
                        match = False
                        if all_satisfy:
                            # 如果all_satisfy为True，则所有非空的include_file_name都必须匹配
                            match = (is_none_or_empty(include_file_name1) or include_file_name1 in file_name_with_ext) and \
                                (is_none_or_empty(include_file_name2) or include_file_name2 in file_name_with_ext) and \
                                (is_none_or_empty(include_file_name3) or include_file_name3 in file_name_with_ext)
                        else:
                            # 如果all_satisfy为False，则至少一个非空的include_file_name需要匹配
                            match = (not is_none_or_empty(include_file_name1) and include_file_name1 in file_name_with_ext) or \
                                (not is_none_or_empty(include_file_name2) and include_file_name2 in file_name_with_ext) or \
                                (not is_none_or_empty(include_file_name3) and include_file_name3 in file_name_with_ext)

                        if match:
                            # 获取文件的各种路径和名称
                            full_path_with_ext = entry.path
                            file_name_without_ext, _ = os.path.splitext(file_name_with_ext)
                            folder_name = os.path.basename(directory)
                            if(check_have_file(os.path.join(check_desdir_havefile,folder_name),file_name_without_ext+file_add_name)):
                                continue
                            current_index += 1
                            if start_index<current_index:
                                # 增量图像计数
                                image_count += 1
                                if entry.name.lower().endswith(tuple(valid_extensions)):
                                    i = Image.open(entry.path)
                                    i = ImageOps.exif_transpose(i)
                                    image = i.convert("RGB")
                                    image = np.array(image).astype(np.float32) / 255.0
                                    image = torch.from_numpy(image)[None,]
                                    images.append(image)

                                parent_folder_name = os.path.basename(os.path.dirname(full_path_with_ext))

                                # 添加到列表中
                                parent_folder_names.append(parent_folder_name)
                                full_paths_with_ext.append(full_path_with_ext)
                                full_paths_without_ext.append(os.path.join(os.path.dirname(full_path_with_ext), file_name_without_ext))
                                file_names_with_ext.append(file_name_with_ext)
                                file_names_without_ext.append(file_name_without_ext)


        except Exception as e:
            # 捕获并处理其他潜在的异常（比如权限问题）
            raise e

        # 如果没有找到文件，则抛出错误
        if not full_paths_with_ext:
            raise FileNotFoundError(f"No files found in directory '{directory}' starting from index {start_index}.")

        return images,parent_folder_names,full_paths_with_ext, full_paths_without_ext, file_names_with_ext, file_names_without_ext, image_count


# 注册节点
NODE_CLASS_MAPPINGS = {
    "LoadFolderFromDir": LoadFolderFromDirList,
    "LoadFilesFromDir": LoadFilesFromDirList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFolderFromDir": "获取子文件夹数据",
    "LoadFilesFromDir": "获取文件夹下文件数据"
}

