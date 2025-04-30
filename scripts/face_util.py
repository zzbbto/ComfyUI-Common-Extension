import cv2
import numpy as np
import os
from typing import Optional

from ..import script_directory

class FaceCheckDetectorNode:
    def __init__(self):
        # 初始化DNN模型相关参数
        self.net: Optional[cv2.dnn_Net] = None
        self.current_prototxt: Optional[str] = None
        self.current_caffemodel: Optional[str] = None

        # 初始化Haar分类器缓存
        self.haar_cascades = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector_type": (["haar", "dnn"], {"default": "haar"}),
            },
            "optional": {
                "detection_target": (
                    ["human", "animal"], 
                    {
                        "default": "human",
                        "display_if": {"detector_type": "haar"}
                    }
                ),
                "scale_factor": (
                    "FLOAT", 
                    {
                        "default": 1.3, 
                        "min": 1.01, 
                        "max": 2.0, 
                        "step": 0.01,
                        "display_if": {"detector_type": "haar"}
                    }
                ),
                "min_neighbors": (
                    "INT", 
                    {
                        "default": 5, 
                        "min": 1, 
                        "max": 10,
                        "display_if": {"detector_type": "haar"}
                    }
                ),
                "min_size": (
                    "INT", 
                    {
                        "default": 30, 
                        "min": 20, 
                        "max": 100,
                        "display_if": {"detector_type": "haar"}
                    }
                ),
                "confidence_threshold": (
                    "FLOAT", 
                    {
                        "default": 0.5, 
                        "min": 0.1, 
                        "max": 1.0, 
                        "step": 0.01,
                        "display_if": {"detector_type": "dnn"}
                    }
                ),
                "prototxt_path": (
                    "STRING", 
                    {
                        "default": "models/deploy.prototxt",
                        "display_if": {"detector_type": "dnn"}
                    }
                ),
                "caffemodel_path": (
                    "STRING", 
                    {
                        "default": "models/res10_300x300_ssd_iter_140000.caffemodel",
                        "display_if": {"detector_type": "dnn"}
                    }
                ),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("has_face", "face_count")
    FUNCTION = "detect_faces"
    CATEGORY = "CommonExtension/face"

    def detect_faces(self, image, detector_type, detection_target="human", scale_factor=1.3, min_neighbors=5, 
                    min_size=30, confidence_threshold=0.5, 
                    prototxt_path="models/deploy.prototxt", 
                    caffemodel_path="models/res10_300x300_ssd_iter_140000.caffemodel"):
        
        # 处理输入图像
        image_np = image[0].cpu().numpy() * 255
        image_np = image_np.astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        h, w = image_np.shape[:2]

        faces = []
        
        if detector_type == "haar":
            # 处理Haar分类器
            cascade_mapping = {
                "human": "haarcascade_frontalface_default.xml",
                "animal": "haarcascade_frontalcatface.xml"
            }
            cascade_file = cascade_mapping.get(detection_target)

            # 缓存分类器
            if cascade_file not in self.haar_cascades:
                cascade_path = os.path.join(cv2.data.haarcascades, cascade_file)
                if not os.path.exists(cascade_path):
                    raise FileNotFoundError(f"Haar分类器文件不存在: {cascade_path}")
                
                classifier = cv2.CascadeClassifier(cascade_path)
                if classifier.empty():
                    raise ValueError(f"无法加载Haar分类器: {cascade_file}")
                self.haar_cascades[cascade_file] = classifier

            # 执行检测
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascades[cascade_file].detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_size, min_size)
            )

        elif detector_type == "dnn":
            # 处理DNN模型路径
            current_prototxt = os.path.join(script_directory, prototxt_path)
            current_caffemodel = os.path.join(script_directory, caffemodel_path)

            # 检查模型路径是否存在
            if not os.path.exists(current_prototxt):
                raise FileNotFoundError(f"Prototxt文件不存在: {current_prototxt}")
            if not os.path.exists(current_caffemodel):
                raise FileNotFoundError(f"Caffemodel文件不存在: {current_caffemodel}")

            # 动态加载模型（仅在路径变化时重新加载）
            if (self.net is None or 
                self.current_prototxt != current_prototxt or 
                self.current_caffemodel != current_caffemodel):
                
                self.release_models()  # 先释放旧模型引用
                try:
                    self.net = cv2.dnn.readNetFromCaffe(current_prototxt, current_caffemodel)
                    self.current_prototxt = current_prototxt
                    self.current_caffemodel = current_caffemodel
                except Exception as e:
                    raise ValueError(f"加载DNN模型失败: {str(e)}")

            # 执行DNN检测
            blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 1.0, 
                                       (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    width = endX - startX
                    height = endY - startY
                    if width >= min_size and height >= min_size:
                        faces.append((startX, startY, width, height))

        has_face = len(faces) > 0
        self.release_models()
        return (has_face, len(faces))

    def release_models(self):
        """释放DNN模型引用，由垃圾回收处理资源"""
        self.net = None
        self.current_prototxt = None
        self.current_caffemodel = None

NODE_CLASS_MAPPINGS = {
    # "FaceCheckDetectorNode": FaceCheckDetectorNode
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    # "FaceCheckDetectorNode": "Face Check Detector Node"
    }