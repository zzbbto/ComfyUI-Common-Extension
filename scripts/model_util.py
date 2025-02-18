import folder_paths
import json
import comfy.samplers

class ParameterTypeJsonLiteral:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": ("STRING", {
                    "default": "sampler"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("list",)
    FUNCTION = "parse_literal"

    CATEGORY = "CommonExtension/model"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    def parse_literal(self, type):
        if type == "test":
            return ("",)
        elif type == "sampler":
            return (comfy.samplers.KSampler.SAMPLERS,)
        elif type == "scheduler":
            return (comfy.samplers.KSampler.SCHEDULERS,)   
        literal = folder_paths.get_filename_list(type)     
        return (literal,)   

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ParameterTypeJsonLiteral": ParameterTypeJsonLiteral,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParameterTypeJsonLiteral": "获取各类参数列表名字",
}

