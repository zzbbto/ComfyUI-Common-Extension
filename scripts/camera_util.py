import random
import numpy as np
from typing import Optional
import math
from collections import OrderedDict
import copy

class CameraCtrlAlluringShot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "zoom_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pan_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "body_focus": (["upper", "full", "face"], {"default": "upper"}),
                "speed_curve": (["linear", "slow_start", "slow_end"], {"default": "slow_end"}),
                "frame_length": ("INT", {"default": 81, "min": 1, "max": 256}),
            },
            "optional": {
                "prev_poses": ("CAMERACTRL_POSES",),
            }
        }

    RETURN_TYPES = ("CAMERACTRL_POSES",)
    CATEGORY = "CommonExtension/camera"
    FUNCTION = "generate_alluring"

    def generate_alluring(self, seed: int, zoom_strength: float, pan_strength: float,
                        body_focus: str, speed_curve: str, frame_length: int,
                        prev_poses: Optional[list]=None):
        # 固定参数配置
        motion_config = {
            "upper": [CAM.ZOOM_IN, CAM.PAN_LEFT, CAM.PAN_RIGHT, CAM.TILT_UP, CAM.ROLL_ANTICLOCKWISE],
            "full": [CAM.ZOOM_OUT, CAM.PAN_DOWN, CAM.TILT_DOWN, CAM.ROLL_CLOCKWISE],
            "face": [CAM.ZOOM_IN, CAM.PAN_LEFT, CAM.PAN_RIGHT, CAM.TILT_UP]
        }

        # 初始化随机种子
        random.seed(seed)
        np.random.seed(seed)

        # 生成运动组合
        motions = []
        strengths = []
        for _ in range(6):
            motion_type = random.choice(motion_config[body_focus])
            
            # 根据运动类型调整强度
            if motion_type in [CAM.ZOOM_IN, CAM.ZOOM_OUT]:
                strength = zoom_strength * random.uniform(0.8, 1.2)
            elif "PAN" in motion_type:
                strength = pan_strength * random.uniform(0.7, 1.3)
            else:
                strength = pan_strength * 0.5
                
            motions.append(motion_type)
            strengths.append(strength)

        # 应用速度曲线
        speed_multiplier = {
            "linear": 1.0,
            "slow_start": 0.6,
            "slow_end": 0.8
        }[speed_curve]

        # 生成相机轨迹
        combined_motion = CameraMotion.combine([
            CAM.get(motions[i]).multiply(strengths[i]) for i in range(6)
        ])
        
        RT = get_camera_motion(
            combined_motion.rotate,
            combined_motion.translate,
            speed=speed_multiplier,
            n=frame_length,
            base=frame_length
        )
        
        new_poses = ndarray_to_poses(RT)
        
        if prev_poses is not None:
            new_poses = combine_poses(prev_poses, new_poses)

        return (new_poses,)

NODE_CLASS_MAPPINGS = {
    "CameraCtrlAlluringShot": CameraCtrlAlluringShot
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraCtrlAlluringShot": "CameraCtrlAlluringShot"
}

class CameraMotion:
    def __init__(self, rotate: tuple[float], translate: tuple[float]):
        assert len(rotate) == 3
        assert len(translate) == 3
        self.rotate = np.array(rotate)
        self.translate = np.array(translate)

    def multiply(self, mult: float):
        if math.isclose(mult, 1.0):
            return self.clone()
        new_rotate = self.rotate.copy()
        new_translate = self.translate.copy()
        new_rotate *= mult
        new_translate *= mult
        return CameraMotion(rotate=new_rotate, translate=new_translate)

    def clone(self):
        return CameraMotion(rotate=self.rotate.copy(), translate=self.translate.copy())

    @staticmethod
    def combine(deltas: list['CameraMotion']) -> 'CameraMotion':
        new_rotate = np.array([0., 0., 0.])
        new_translate = np.array([0., 0., 0.])
        for delta in deltas:
            new_rotate += delta.rotate
            new_translate += delta.translate
        return CameraMotion(rotate=new_rotate, translate=new_translate)

class CAM:
    BASE_T_NORM = 1.5
    BASE_ANGLE = np.pi/3

    DEFAULT_FX = 0.474812461
    DEFAULT_FY = 0.844111024
    DEFAULT_CX = 0.5
    DEFAULT_CY = 0.5

    DEFAULT_POSE_WIDTH = 1280
    DEFAULT_POSE_HEIGHT = 720

    STATIC = "Static"
    PAN_UP = "Pan Up" #向上移动
    PAN_DOWN = "Pan Down"
    PAN_LEFT = "Pan Left"
    PAN_RIGHT = "Pan Right"
    ZOOM_IN = "Zoom In" #拉近
    ZOOM_OUT = "Zoom Out"
    ROLL_CLOCKWISE = "Roll Clockwise" #顺时针旋转
    ROLL_ANTICLOCKWISE = "Roll Anticlockwise"
    TILT_UP = "Tilt Up" #向上倾斜
    TILT_DOWN = "Tilt Down"
    TILT_LEFT = "Tilt Left"
    TILT_RIGHT = "Tilt Right"
    
    _PAIRS = [
        (STATIC,        CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., 0.))),
        (PAN_UP,        CameraMotion(rotate=(0., 0., 0.), translate=(0., 1., 0.))),
        (PAN_DOWN,      CameraMotion(rotate=(0., 0., 0.), translate=(0., -1., 0.))),
        (PAN_LEFT,      CameraMotion(rotate=(0., 0., 0.), translate=(1., 0., 0.))),
        (PAN_RIGHT,     CameraMotion(rotate=(0., 0., 0.), translate=(-1., 0., 0.))),
        (ZOOM_IN,       CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., -2.))),
        (ZOOM_OUT,      CameraMotion(rotate=(0., 0., 0.), translate=(0., 0., 2.))),
        (ROLL_CLOCKWISE,     CameraMotion(rotate=(0., 0., -1.), translate=(0., 0., 0.))),
        (ROLL_ANTICLOCKWISE, CameraMotion(rotate=(0., 0., 1.), translate=(0., 0., 0.))),
        (TILT_DOWN,     CameraMotion(rotate=(1., 0., 0.), translate=(0., 0., 0.))),
        (TILT_UP,    CameraMotion(rotate=(-1., 0., 0.), translate=(0., 0., 0.))),
        (TILT_LEFT,       CameraMotion(rotate=(0., 1., 0.), translate=(0., 0., 0.))),
        (TILT_RIGHT,     CameraMotion(rotate=(0., -1., 0.), translate=(0., 0., 0.))),
    ]
    _DICT: dict[str, CameraMotion] = OrderedDict(_PAIRS)
    _LIST = list(_DICT.keys())

    @staticmethod
    def get(motion: str):
        return CAM._DICT[motion]
    
def compute_R_from_rad_angle(angles: np.ndarray):
    theta_x, theta_y, theta_z = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def get_camera_motion(angle: np.ndarray, T: np.ndarray, speed: float, n=16, base=16):
    RT = []
    for i in range(n):
        _angle = (i/base)*speed*(CAM.BASE_ANGLE)*angle
        R = compute_R_from_rad_angle(_angle) 
        # _T = (i/n)*speed*(T.reshape(3,1))
        _T=(i/base)*speed*(CAM.BASE_T_NORM)*(T.reshape(3,1))
        _RT = np.concatenate([R,_T], axis=1)
        RT.append(_RT)
    RT = np.stack(RT)
    return RT    

def ndarray_to_poses(RT: np.ndarray, fx=CAM.DEFAULT_FX, fy=CAM.DEFAULT_FY, cx=CAM.DEFAULT_CX, cy=CAM.DEFAULT_CY) -> list[list[float]]:
    '''
    Converts ndarray (motion) to cameractrl_poses.
    '''
    motion_list=RT.tolist()
    poses = []
    for motion in motion_list:
        traj = [0, fx, fy, cx, cy, CAM.DEFAULT_POSE_WIDTH, CAM.DEFAULT_POSE_HEIGHT]
        traj.extend(motion[0])
        traj.extend(motion[1])
        traj.extend(motion[2])
        poses.append(traj)
    return poses

def combine_poses(poses0: list[list[float]], poses1: list[list[float]]):
    new_poses = copy.deepcopy(poses0) + copy.deepcopy(poses1)
    new_RT = combine_RTs(poses_to_ndarray(poses0), poses_to_ndarray(poses1))
    inter_poses = ndarray_to_poses(new_RT)
    # maintain fx, fy, cx, and cy values by pasting only the movement portion of poses
    for i in range(len(new_poses)):
        new_poses[i][7:] = inter_poses[i][7:]
    return new_poses

def combine_RTs(RT_0: np.ndarray, RT_1: np.ndarray):
    RT = copy.deepcopy(RT_0[-1])
    R = RT[:,:3]
    R_inv = RT[:,:3].T
    T =  RT[:,-1]

    temp = []
    for _RT in RT_1:
        _RT[:,:3] = np.dot(_RT[:,:3], R)
        _RT[:,-1] =  _RT[:,-1] + np.dot(np.dot(_RT[:,:3], R_inv), T) 
        temp.append(_RT)

    RT_1 = np.stack(temp)

    return np.concatenate([RT_0, RT_1], axis=0)

def poses_to_ndarray(poses: list[list[float]]) -> np.ndarray:
    '''
    Converts cameractrl_poses (list) to ndarray (motion) to be used for math stuff.
    '''
    motion_list = []
    for pose in poses:
        # pose will have 19 components;
        # idx 7-10 have first column, idx 11-14 have second column, idx 15-18 have third column
        motion_list.append(np.array(pose[7:]).reshape(3, 4))
    RT = np.array(motion_list)
    return RT