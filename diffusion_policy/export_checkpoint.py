#-*- coding:utf-8 -*-
#
# Export Diffusion Policy checkpoint to PyTorch Models
#

import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dill
import torch
import argparse
from dataset.tasks import ControlType, PushT, TaskTags, TaskTypes
from diffusion.edm_model import MultiModelType

from diffusion.model import ConditionalUnet1D, get_diffusion_policy_obs_encoder, get_resnet, replace_bn_with_gn
from diffusion.transformer import TransformerForDiffusion 

CONTROL_TYPE = ControlType.IMAGE

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--diffusion_policy_checkpoint', type=str, default="./weights/diffusion_policy_cnn_image_pusht_latest.ckpt")
parser.add_argument('--task_type', type=str, default="PUSHT")
parser.add_argument('--task_tag', type=str, default="")
opt = parser.parse_args()

if opt.task_type == 'PUSHT':
    task_type = TaskTypes.PUSHT
    task = PushT(ctype=CONTROL_TYPE)
elif opt.task_type == 'LIFT':
    task_type = TaskTypes.LIFT
elif opt.task_type == 'CAN':
    task_type = TaskTypes.CAN
elif opt.task_type == 'SQUARE':
    task_type = TaskTypes.SQUARE
elif opt.task_type == 'TRANSPORT':
    task_type = TaskTypes.TRANSPORT
elif opt.task_type == 'TOOLHANG':
    task_type = TaskTypes.TOOLHANG
else:
    raise NotImplementedError(f"Task {opt.task_type} Not implemented")

if opt.task_tag == "":
    task_tag = TaskTags.NONE
elif opt.task_tag == "PH":
    task_tag = TaskTags.PH 
elif opt.task_tag == "MH":
    task_tag = TaskTags.MH
else:
    raise NotImplementedError(f"Task Tag {opt.task_tag} Not implemented")

def load_model(model_type:MultiModelType = MultiModelType.CNN):
    action_dim = task.action_dim
    # dim = opt.image_encode_dim + opt.obs_dim
    dim = task.obs_dim
    device = torch.device('cuda')

    if model_type == MultiModelType.CNN:
        ddpm = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=dim*task.obs_horizon,
            diffusion_step_embed_dim=128
        )
    elif model_type == MultiModelType.MINGPT:
        ddpm = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=task.pred_horizon,
            n_obs_steps=task.obs_horizon,
            cond_dim=dim,
            causal_attn=True,
            n_cond_layers=4
        )
    else:
        raise NotImplementedError

    # vision_encoder = get_resnet('resnet18', image_encode_dim=task.image_encode_dim)
    # vision_encoder = replace_bn_with_gn(vision_encoder).to(device)
    vision_encoder = get_diffusion_policy_obs_encoder(task).to(device)
    vision_encoder.eval()

    nets = torch.nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': ddpm
    })
    print('Pretrained DDPM model loaded.')
    return nets

def get_obs_encoder(state_dict):
    encoder_key_num = len([key for key in state_dict.keys() if "obs_encoder" in key])
    encoder_weights = {}
    for key in state_dict:
        if "obs_encoder" in key:
            new_key = key.replace("obs_encoder.", "")
            encoder_weights[new_key] = state_dict[key]
    return encoder_weights

def get_inner_model(state_dict):
    key_num = len([key for key in state_dict.keys() if "model." in key])
    weights = {}
    for key in state_dict:
        if "model." in key:
            new_key = key.replace("model.", "")
            weights[new_key] = state_dict[key]
    return weights

def main():
    device = 'cuda'
    # model = load_model()
    # model_state_dict = model.state_dict()
    # inner_model_state_dict = model['noise_pred_net'].state_dict()

    diffusion_policy_state_dict = torch.load(open(opt.diffusion_policy_checkpoint, 'rb'), pickle_module=dill)
    # diffusion_policy_model_state_dict = diffusion_policy_state_dict["state_dicts"]["model"]
    diffusion_policy_model_state_dict = diffusion_policy_state_dict["state_dicts"]["ema_model"]

    # print("1>", len(model_state_dict), len(diffusion_policy_model_state_dict))
    vision_encoder = get_diffusion_policy_obs_encoder(task).to(device)
    vision_encoder.eval()

    encoder_weights = get_obs_encoder(diffusion_policy_model_state_dict)
    vision_encoder.load_state_dict(encoder_weights)
    print("Encoder Loaded!")

    inner_weights = get_inner_model(diffusion_policy_model_state_dict)
    ddpm = ConditionalUnet1D(
        input_dim=task.action_dim,
        global_cond_dim=task.obs_dim*task.obs_horizon,
        diffusion_step_embed_dim=128
    )
    ddpm.load_state_dict(inner_weights)
    model = torch.nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': ddpm
    })
    torch.save(model.state_dict(), 'exported_diffusion_policy_latest.ckpt')
    print("EXPORTED!")


if __name__ == '__main__':
    main()