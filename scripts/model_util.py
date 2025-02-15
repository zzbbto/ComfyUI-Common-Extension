import folder_paths
import json


def GetModelJsonFormLiteral(literal):
    # literal = "\n".join(folder_paths.get_filename_list(literal))
    literal = folder_paths.get_filename_list(literal)
    # # 分割并过滤空行
    # split = list(filter(None, literal.split("\n")))
    # # 转换为包含name字段的字典列表
    # json_list = [{"name": item.strip()} for item in split if item.strip()]
    # # 序列化为JSON字符串
    # json_str = json.dumps(json_list, indent=2)

    # literal = folder_paths.get_filename_list(literal)
    # literal = json.dumps(literal)
    return literal

class ModelTypeJsonLiteral:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelType": ("STRING", {
                    "default": "checkpoints"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "parse_literal"

    CATEGORY = "CommonExtension/model"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    def parse_literal(self, modelType):
        # 序列化为JSON字符串
        json_str = GetModelJsonFormLiteral(modelType)
        return (json_str,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ModelTypeJsonLiteral": ModelTypeJsonLiteral,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelTypeJsonLiteral": "获取各类模型列表名字json",
}

