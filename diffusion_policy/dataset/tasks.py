#-*- coding:utf-8 -*-

from typing import List
from enum import Enum 

class ControlType(Enum):
    STATE = "STATE"
    IMAGE = "IMAGE"

class TaskTypes(Enum):
    PUSHT = "PUSHT"
    LIFT = "LIFT"
    CAN = "CAN"
    SQUARE = "SQUARE"
    TRANSPORT = "TRANSPORT"
    TOOLHANG = "TOOLHANG"

class TaskTags(Enum):
    NONE = "NONE"
    PH = "PH"
    MH = "MH"

class PushT:
    def __init__(self, ctype:ControlType = ControlType.IMAGE) -> None:
        self.ctype = ctype
        self.dataset_path= "./data/pusht/pusht_cchi_v7_replay.zarr.zip"
        if ctype == ControlType.STATE:
            self.pred_horizon: int = 16  # "time steps of predicted actions"
            self.action_horizon: int = 8 
            self.obs_horizon: int = 2 # "input observation condition size of timeseires"
            self.action_dim: int = 2 # "action dimension size : T-Push = (x, y) so that it's 2"
            self.obs_dim: int = 5 # "observation dimension size : T-Push low -> 2"
        else:
            self.pred_horizon: int = 16 
            self.action_horizon: int = 8
            self.obs_horizon: int = 2
            self.action_dim: int = 2 
            self.image_encode_dim = 64 # 512
            self.low_dim = 2
            self.obs_dim: int = self.image_encode_dim + self.low_dim
            self.image_shape: list = [3, 96, 96]
    
    def get_eval_seeds(self, test_samples:int=50, train_samples:int=6):
        train_start_seed = 0
        test_start_seed = 10000
        env_seeds = [train_start_seed + i for i in range(train_samples)]
        env_seeds += [test_start_seed + i for i in range(test_samples)]
        return env_seeds

    def get_shape_meta(self):
        return {}

class Can:
    def __init__(self, ctype:ControlType = ControlType.IMAGE, tag:TaskTags=TaskTags.MH) -> None:
        self.ctype = ctype
        self.tag = tag
        self.bottle_neck_periods = []
        if ctype == ControlType.STATE:
            self.dataset_path= f"./dataset/robomimic/can/mh/low_dim_abs.hdf5" if tag == TaskTags.MH else f"./dataset/robomimic/can/ph/low_dim_abs.hdf5"
            self.pred_horizon: int = 16  
            self.action_horizon: int = 8 
            self.obs_horizon: int = 2 
            self.action_dim: int = 10 
            self.obs_dim: int = 23
            self.obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos'
            ]
        else:
            self.dataset_path= f"./dataset/robomimic/can/mh/image_abs.hdf5" if tag == TaskTags.MH else f"./dataset/robomimic/can/ph/image_abs.hdf5"
            self.pred_horizon: int = 16 
            self.action_horizon: int = 8
            self.obs_horizon: int = 2
            self.action_dim: int = 10 
            self.image_encode_dim = 64 # 512
            self.agaent_view_image_shape: list = [3, 84, 84]
            self.robot0_eef_pos = 3
            self.robot0_eef_quat = 4
            self.robot0_eye_in_hand_image_shape = [3, 84, 84]
            self.robot0_gripper_qpos = 2
            self.obs_dim: int = self.image_encode_dim * 2 + self.robot0_eef_pos + self.robot0_eef_quat + self.robot0_gripper_qpos # = 137
    
    def get_eval_seeds(self, test_samples:int=50, train_samples:int=6):
        train_start_seed = 0
        test_start_seed = 100000
        env_seeds = [train_start_seed + i for i in range(train_samples)]
        env_seeds += [test_start_seed + i for i in range(test_samples)]
        return env_seeds
    
    def get_shape_meta(self):
        return {
            'action': 
                {
                    'shape' : [self.action_dim,]
                },
            'obs':
                {
                    'agentview_image':{
                        'shape': self.agaent_view_image_shape,
                        'type': 'rgb'
                    },
                    'robot0_eef_pos':{
                        'shape': [self.robot0_eef_pos,]
                    },
                    'robot0_eef_quat':{
                        'shape': [self.robot0_eef_quat,]
                    },
                    'robot0_eye_in_hand_image':{
                        'shape':self.robot0_eye_in_hand_image_shape,
                        'type':'rgb'
                    },
                    'robot0_gripper_qpos':{
                        'shape': [self.robot0_gripper_qpos,]
                    }
                }, 
        }
    
class Lift:
    def __init__(self, ctype:ControlType = ControlType.IMAGE, tag:TaskTags=TaskTags.MH) -> None:
        self.ctype = ctype
        self.tag = tag
        self.bottle_neck_periods = []
        if ctype == ControlType.STATE:
            self.dataset_path= f"./data/robomimic/datasets/lift/mh/low_dim_abs.hdf5" if tag == TaskTags.MH else f"./data/robomimic/datasets/lift/ph/low_dim_abs.hdf5"
            self.pred_horizon: int = 16  
            self.action_horizon: int = 8 
            self.obs_horizon: int = 2 
            self.action_dim: int = 10 
            self.obs_dim: int = 19
            self.obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos'
            ]
        else:
            self.dataset_path= f"./data/robomimic/datasets/lift/mh/image_abs.hdf5" if tag == TaskTags.MH else f"./data/robomimic/datasets/lift/ph/image_abs.hdf5"
            self.pred_horizon: int = 16 
            self.action_horizon: int = 8
            self.obs_horizon: int = 2
            self.action_dim: int = 10 
            self.image_encode_dim = 512
            self.agaent_view_image_shape: list = [3, 84, 84]
            self.robot0_eef_pos = 3
            self.robot0_eef_quat = 4
            self.robot0_eye_in_hand_image_shape = [3, 84, 84]
            self.robot0_gripper_qpos = 2
            self.obs_dim: int = self.image_encode_dim * 2 + self.robot0_eef_pos + self.robot0_eef_quat + self.robot0_gripper_qpos # = 137
    
    def get_eval_seeds(self, test_samples:int=50, train_samples:int=6):
        train_start_seed = 0
        test_start_seed = 10000
        env_seeds = [train_start_seed + i for i in range(train_samples)]
        env_seeds += [test_start_seed + i for i in range(test_samples)]
        return env_seeds
    
    def get_shape_meta(self):
        return {
            'action': 
                {
                    'shape' : [self.action_dim,]
                },
            'obs':
                {
                    'agentview_image':{
                        'shape': self.agaent_view_image_shape,
                        'type': 'rgb'
                    },
                    'robot0_eef_pos':{
                        'shape': [self.robot0_eef_pos,]
                    },
                    'robot0_eef_quat':{
                        'shape': [self.robot0_eef_quat,]
                    },
                    'robot0_eye_in_hand_image':{
                        'shape':self.robot0_eye_in_hand_image_shape,
                        'type':'rgb'
                    },
                    'robot0_gripper_qpos':{
                        'shape': [self.robot0_gripper_qpos,]
                    }
                }, 
        }

class Square:
    def __init__(self, ctype:ControlType = ControlType.IMAGE, tag:TaskTags=TaskTags.MH) -> None:
        self.ctype = ctype
        self.tag = tag
        self.bottle_neck_periods = []
        if ctype == ControlType.STATE:
            self.dataset_path= f"./dataset/robomimic/square/mh/low_dim_abs.hdf5" if tag == TaskTags.MH else f"./dataset/robomimic/square/ph/low_dim_abs.hdf5"
            self.pred_horizon: int = 16  
            self.action_horizon: int = 8 
            self.obs_horizon: int = 2 
            self.action_dim: int = 10 
            self.obs_dim: int = 23
            self.obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos'
            ]
        else:
            self.dataset_path= f"./dataset/robomimic/square/mh/image_abs.hdf5" if tag == TaskTags.MH else f"./dataset/robomimic/square/ph/image_abs.hdf5"
            self.pred_horizon: int = 16 
            self.action_horizon: int = 8
            self.obs_horizon: int = 2
            self.action_dim: int = 10 
            self.image_encode_dim = 512
            self.agaent_view_image_shape: list = [3, 84, 84]
            self.robot0_eef_pos = 3
            self.robot0_eef_quat = 4
            self.robot0_eye_in_hand_image_shape = [3, 84, 84]
            self.robot0_gripper_qpos = 2
            self.obs_dim: int = self.image_encode_dim * 2 + self.robot0_eef_pos + self.robot0_eef_quat + self.robot0_gripper_qpos # = 137
    
    def get_eval_seeds(self, test_samples:int=50, train_samples:int=6):
        train_start_seed = 0
        test_start_seed = 10000
        env_seeds = [train_start_seed + i for i in range(train_samples)]
        env_seeds += [test_start_seed + i for i in range(test_samples)]
        return env_seeds
    
    def get_shape_meta(self):
        return {
            'action': 
                {
                    'shape' : [self.action_dim,]
                },
            'obs':
                {
                    'agentview_image':{
                        'shape': self.agaent_view_image_shape,
                        'type': 'rgb'
                    },
                    'robot0_eef_pos':{
                        'shape': [self.robot0_eef_pos,]
                    },
                    'robot0_eef_quat':{
                        'shape': [self.robot0_eef_quat,]
                    },
                    'robot0_eye_in_hand_image':{
                        'shape':self.robot0_eye_in_hand_image_shape,
                        'type':'rgb'
                    },
                    'robot0_gripper_qpos':{
                        'shape': [self.robot0_gripper_qpos,]
                    }
                }, 
        }
    

class ToolHang:
    def __init__(self, ctype:ControlType = ControlType.IMAGE) -> None:
        self.ctype = ctype
        self.bottle_neck_periods = [
            [200, 250], [300, 350]
        ]
        if ctype == ControlType.STATE:
            self.dataset_path= f"./data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5"
            # self.dataset_path= f"./low_dim_abs-edm1.hdf5"
            self.pred_horizon: int = 16  
            self.action_horizon: int = 8 
            self.obs_horizon: int = 2 
            self.action_dim: int = 10 
            self.obs_dim: int = 53
            self.obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos'
            ]
        else:
            self.dataset_path= f"./data/robomimic/datasets/tool_hang/ph/image_abs.hdf5"
            self.pred_horizon: int = 16 
            self.action_horizon: int = 8
            self.obs_horizon: int = 2
            self.action_dim: int = 10 
            self.image_encode_dim = 64 # 512
            self.sideview_image: list = [3, 240, 240]
            self.robot0_eef_pos = 3
            self.robot0_eef_quat = 4
            self.robot0_eye_in_hand_image_shape = [3, 240, 240]
            self.robot0_gripper_qpos = 2
            self.obs_dim: int = self.image_encode_dim * 2 + self.robot0_eef_pos + self.robot0_eef_quat + self.robot0_gripper_qpos # = 137
    
    def get_eval_seeds(self, test_samples:int=50, train_samples:int=6):
        train_start_seed = 0
        test_start_seed = 10000
        env_seeds = [train_start_seed + i for i in range(train_samples)]
        env_seeds += [test_start_seed + i for i in range(test_samples)]
        return env_seeds
    
    def get_shape_meta(self):
        return {
            'action': 
                {
                    'shape' : [self.action_dim,]
                },
            'obs':
                {
                    'sideview_image':{
                        'shape': self.sideview_image,
                        'type': 'rgb'
                    },
                    'robot0_eef_pos':{
                        'shape': [self.robot0_eef_pos,]
                    },
                    'robot0_eef_quat':{
                        'shape': [self.robot0_eef_quat,]
                    },
                    'robot0_eye_in_hand_image':{
                        'shape':self.robot0_eye_in_hand_image_shape,
                        'type':'rgb'
                    },
                    'robot0_gripper_qpos':{
                        'shape': [self.robot0_gripper_qpos,]
                    }
                }, 
        }

class Transport:
    def __init__(self, ctype:ControlType = ControlType.IMAGE, tag:TaskTags=TaskTags.MH) -> None:
        self.ctype = ctype
        if tag == TaskTags.MH:
            self.bottle_neck_periods = [
                [200, 400]
            ] # 
        else:
            self.bottle_neck_periods = [
                [200, 400]
            ] # 
        if ctype == ControlType.STATE:
            self.dataset_path= f"./data/robomimic/datasets/transport/mh/low_dim_abs.hdf5" if tag == TaskTags.MH else f"./data/robomimic/datasets/transport/ph/low_dim_abs.hdf5"
            self.pred_horizon: int = 16  
            self.action_horizon: int = 8 
            self.obs_horizon: int = 2 
            self.action_dim: int = 20 
            self.obs_dim: int = 59
            self.obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos',
                'robot1_eef_pos',
                'robot1_eef_quat',
                'robot1_gripper_qpos'
            ]
        else:
            self.dataset_path= f"./data/robomimic/datasets/transport/mh/image_abs.hdf5" if tag == TaskTags.MH else f"./data/robomimic/datasets/transport/ph/image_abs.hdf5"
            self.pred_horizon: int = 16 
            self.action_horizon: int = 8
            self.obs_horizon: int = 2
            self.action_dim: int = 20 
            self.image_encode_dim = 64 # 512
            self.robot0_eef_pos = 3
            self.robot0_eef_quat = 4
            self.robot0_eye_in_hand_image_shape = [3, 84, 84]
            self.robot0_gripper_qpos = 2
            self.robot1_eef_pos = 3
            self.robot1_eef_quat = 4
            self.robot1_eye_in_hand_image = [3, 84, 84]
            self.robot1_gripper_qpos = 2
            self.shouldercamera0_image = [3, 84, 84]
            self.shouldercamera1_image = [3, 84, 84]
            self.obs_dim: int = self.image_encode_dim * 4 \
                    + self.robot0_eef_pos + self.robot0_eef_quat + self.robot0_gripper_qpos \
                    + self.robot1_eef_pos + self.robot1_eef_quat + self.robot1_gripper_qpos
    
    def get_eval_seeds(self, test_samples:int=50, train_samples:int=6):
        train_start_seed = 0
        test_start_seed = 10000
        env_seeds = [train_start_seed + i for i in range(train_samples)]
        env_seeds += [test_start_seed + i for i in range(test_samples)]
        return env_seeds
    
    def get_shape_meta(self):
        return {
            'action': 
                {
                    'shape' : [self.action_dim,]
                },
            'obs':
                {
                    'robot0_eef_pos':{
                        'shape': [self.robot0_eef_pos,]
                    },
                    'robot0_eef_quat':{
                        'shape': [self.robot0_eef_quat,]
                    },
                    'robot0_eye_in_hand_image':{
                        'shape':self.robot0_eye_in_hand_image_shape,
                        'type':'rgb'
                    },
                    'robot0_gripper_qpos':{
                        'shape': [self.robot0_gripper_qpos,]
                    },
                    'robot1_eef_pos':{
                        'shape':[self.robot1_eef_pos,]
                    },
                    'robot1_eef_quat':{
                        'shape':[self.robot1_eef_quat,]
                    },
                    'robot1_eye_in_hand_image':{
                        'shape':self.robot1_eye_in_hand_image,
                        'type':'rgb'
                    },
                    'robot1_gripper_qpos':{
                        'shape':[self.robot1_gripper_qpos,]
                    },
                    'shouldercamera0_image':{
                        'shape':self.shouldercamera0_image,
                        'type': 'rgb'
                    },
                    'shouldercamera1_image':{
                        'shape':self.shouldercamera1_image,
                        'type': 'rgb'
                    }
                }, 
        }