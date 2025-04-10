#-*- coding:utf-8 -*-

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.rotation_transformer import RotationTransformer
# from diffusion_policy.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.dataset.robomimic_image_dataset import RobomimicReplayImageDataset
from IPython.display import Video
from skvideo.io import vwrite
from diffusion_policy.dataset.tasks import *
import wandb.sdk.data_types.video as wv
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from scripts.agilex_model import create_model

import argparse
import collections
import numpy as np
import pathlib
import random 
import torch
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--task-tag', type=str, default="PH")
parser.add_argument('--task-type', type=str, default="CAN")
opt = parser.parse_args()

CONTROL_TYPE = ControlType.IMAGE

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(opt.seed)
device = 'cuda'

if opt.task_tag == "":
    task_tag = TaskTags.NONE
elif opt.task_tag == "PH":
    task_tag = TaskTags.PH 
elif opt.task_tag == "MH":
    task_tag = TaskTags.MH
else:
    raise NotImplementedError(f"Task Tag {opt.task_tag} Not implemented")

if opt.task_type == 'PUSHT':
    task_type = TaskTypes.PUSHT
    task = PushT(ctype=CONTROL_TYPE)
elif opt.task_type == 'LIFT':
    task_type = TaskTypes.LIFT
    task = Lift(ctype=CONTROL_TYPE, tag=task_tag)
elif opt.task_type == 'CAN':
    task_type = TaskTypes.CAN
    task = Can(ctype=CONTROL_TYPE, tag=task_tag)
elif opt.task_type == 'SQUARE':
    task_type = TaskTypes.SQUARE
    task = Square(ctype=CONTROL_TYPE, tag=task_tag)
elif opt.task_type == 'TRANSPORT':
    task_type = TaskTypes.TRANSPORT
    task = Transport(ctype=CONTROL_TYPE, tag=task_tag)
elif opt.task_type == 'TOOLHANG':
    task_type = TaskTypes.TOOLHANG
    task = ToolHang(ctype=CONTROL_TYPE)
else:
    raise NotImplementedError(f"Task {opt.task_type} Not implemented")

def create_env(
        seed:int=0, 
        enable_render:bool=True, 
        output_dir:str="results", 
        render_hw=(256,256),
        render_camera_name="agentview",
        fps:int=10,
        crf:int=22
    ):
    # if task_type == TaskTypes.PUSHT:
    #     env = PushTEnv()
    #     # use a seed >200 to avoid initial states seen in the training dataset
    #     env.seed(seed)
    #     return env 
    if 1:
        env_meta = FileUtils.get_env_metadata_from_dataset(task.dataset_path)
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False
        modality_mapping = collections.defaultdict(list)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        ObsUtils.initialize_obs_modality_mapping_from_dict(
            {'low_dim': task.obs_keys})
        _env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            # only way to not show collision geometry
            # is to enable render_offscreen
            # which uses a lot of RAM.
            render_offscreen=False,
            use_image_obs=False, 
        )
        # Robosuite's hard reset causes excessive memory consumption.
        # Disabled to run more envs.
        # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
        _env.hard_reset = False
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        env = VideoRecordingWrapper(
            RobomimicImageWrapper(
                env=_env,
                obs_keys=task.obs_keys,
                init_state=None,
                render_hw=render_hw,
                render_camera_name=render_camera_name
            ),
            video_recoder=VideoRecorder.create_h264(
            fps=fps,
            codec='h264',
            input_pix_fmt='rgb24',
            crf=crf,
            thread_type='FRAME',
            thread_count=1
            ),
            file_path=None,
            steps_per_render=steps_per_render
        )

        env.video_recoder.stop()
        env.file_path = None
        if enable_render:
            filename = pathlib.Path(output_dir).joinpath(
                'media', opt.task_type + "-" + opt.task_tag + "_" + wv.util.generate_id() + ".mp4")
            filename.parent.mkdir(parents=False, exist_ok=True)
            filename = str(filename)
            env.file_path = filename

        # switch to seed reset
        assert isinstance(env.env, RobomimicImageWrapper)
        env.env.env.init_state = None
        env.seed(seed)
        return env

def undo_transform_action(action, rotation_transformer:RotationTransformer):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1,2,10)

    d_rot = action.shape[-1] - 4
    pos = action[...,:3]
    rot = action[...,3:3+d_rot]
    gripper = action[...,[-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([
            pos, rot, gripper
    ], axis=-1)

    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction

def get_stats():
    dataset = RobomimicReplayImageDataset(
        dataset_path=task.dataset_path,
        horizon=task.pred_horizon,
        obs_keys=task.obs_keys,
        abs_action=True,
        pad_before=1,
        pad_after=7,
    )
    return dataset 

def main():
    stats = get_stats()
    model_card = "robotics-diffusion-transformer/rdt-170m"
    
    # Names of cameras used for visual input
    CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
    device = 'cuda'
    config = {
        'episode_len': 1000,  # Max length of one episode
        'state_dim': 14,      # Dimension of the robot's state
        'chunk_size': 64,     # Number of actions to predict in one step
        'camera_names': CAMERA_NAMES,
    }
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384" 
    tensor_type = torch.bfloat16
    
    model = create_model(
        args=config,
        dtype=tensor_type, 
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        pretrained=model_card,
        control_frequency=25,
        device=device
    )

    torch_model = model.policy.model
    torch_model.eval()
    print("Model Loaded!")
    
    
if __name__ == '__main__':
    main()