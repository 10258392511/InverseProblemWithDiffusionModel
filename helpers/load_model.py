import torch
import yaml
import os
import glob
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from InverseProblemWithDiffusionModel.ncsn.models.ncsnv2 import NCSNv2Deepest, NCSNv2
from InverseProblemWithDiffusionModel.ncsn.models.ncsn1d import NCSN1D
from InverseProblemWithDiffusionModel.ncsn.models.classifiers import ResNetClf
from monai.networks.nets import UNet
from InverseProblemWithDiffusionModel.helpers.load_data import load_config
from InverseProblemWithDiffusionModel.helpers.utils import load_yml_file, collate_state_dict
from InverseProblemWithDiffusionModel.helpers.pl_helpers import TrainScoreModelDiscrete, TrainClf, TrainSeg


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
with open(os.path.join(parent_dir, "ncsn/configs/general_config.yml"), "r") as rf:
    general_config = yaml.load(rf, yaml.Loader)


TASK_NAME_TO_MODEL_CTOR = {
    "Diffusion": NCSNv2Deepest,
    "Diffusion1D": NCSN1D,
    "Clf": ResNetClf,
    "Seg": UNet
}


# RELOAD_ROOT_DIRS = {
#     "Diffusion": r"E:\ncsn_logs",
#     "Clf": r"D:\testings\Python\TestingPython\InverseProblemWithDiffusionModel\clf_logs",
#     "Seg": r"D:\testings\Python\TestingPython\InverseProblemWithDiffusionModel\seg_logs"
# }



RELOAD_ROOT_DIRS = {
    "Diffusion": r"/scratch/zhexwu/InverseProblemWithDiffusionModel/ncsn_logs",
    "Clf": r"/scratch/zhexwu/InverseProblemWithDiffusionModel/clf_logs",
    "Seg": r"/scratch/zhexwu/InverseProblemWithDiffusionModel/seg_logs",
}


RELOAD_MODEL_DIRS = {
    "Diffusion": {
        "MNIST": {
            "real-valued": "2022_11_04_23_30_30_182831",
            "complex": "2022_11_04_23_33_00_959536"
        },
        "CINE64": {
            # "real-valued": "2022_11_04_23_41_05_847127",  # [0, 1]
            # "real-valued": "2022_12_24_09_13_59_663457",  # [-1, 1]
            # "real-valued": "2023_01_05_23_50_36_557715",  # real-imag, [0, 1] -> [-1, 1]
            "real-valued": "2023_02_03_22_37_06_400286",  # real-imag, [0, 1]
            "complex": "2022_11_04_23_42_23_930412"
        },
        "CINE64_1D": {
            # "real-valued": "2023_01_21_11_05_11_826573",  # real-imag, [0, 1]
            "real-valued": "2023_02_03_22_37_54_991247"  # real-imag, [0, 1]
        },
        "CINE127": {
            # "real-valued": "2022_11_04_23_58_37_174974",
            "real-valued": "2023_02_03_22_36_00_374061",  # real-imag, [0, 1]
            "complex": "2022_11_05_00_02_01_120409"
        },
        "CINE127_1D": {
            # "real-valued": "2023_01_21_11_05_27_208935",  # real-imag, [0, 1]
            "real-valued": "2023_02_05_07_46_19_118951"  # real_imag, [0, 1]
        },
        "ACDC": {
            # "real-valued": "2022_11_07_10_48_24_147215",  # [0, 1]
            # "real-valued": "2022_12_24_09_13_41_113794",  # [-1, 1], 200 epochs, B = 4
            # "real-valued": "2022_12_25_11_59_46_710463",  # [-1, 1], 400 epochs, B = 6
            "real-valued": "2023_01_05_23_55_24_713049",  # real-imag, [0, 1] -> [-1, 1]
            "complex": "2022_11_07_10_48_52_039130"
        }
    },

    "Clf": {
        "MNIST": {
            "real-valued": "2022_11_08_00_49_36_533050",
            "complex": "2022_11_08_00_53_31_543770"
        }
    },

    "Seg": {
        "ACDC": {
            # "real-valued": "2022_11_11_23_13_49_676811",  # [0, 1]
            # "real-valued": "2022_12_24_09_08_13_847889",  # [-1, 1]
            # "real-valued": "2023_01_12_00_17_44_526067",  # real-imag, [0, 1] -> [-1, 1]
            "real-valued": "2023_01_14_00_29_38_103272",  # real-imag-random, [0, 1] -> [-1, 1] 
            "complex": "2022_11_06_16_11_52_335747"
        }
    }
}


def load_model(config, task_name, use_net_params=False):
    global general_config
    assert task_name in TASK_NAME_TO_MODEL_CTOR
    net_ctor = TASK_NAME_TO_MODEL_CTOR[task_name]
    if use_net_params:
        net_params = general_config[task_name]
        net_params["in_channels"] = config.data.channels

    model = None
    if "Diffusion" in task_name:
        model = net_ctor(config)

    elif task_name == "Clf":
        model = net_ctor(net_params)

    elif task_name == "Seg":
        model = net_ctor(**net_params)

    return model


def reload_model(task_name, ds_name, mode):
    config = load_config(ds_name, mode=mode, device=ptu.DEVICE)
    mode_out = None
    if "Diffusion" in task_name:
        model = load_model(config, task_name, False)
        model_out = reload_ncsn(model, config, task_name, ds_name)
    elif task_name == "Clf":
        model = load_model(config, task_name, True)
        model_out = reload_clf(model, config, task_name, ds_name)
    elif task_name == "Seg":
        model = load_model(config, task_name, True)
        model_out = reload_seg(model, config, task_name, ds_name)
    else:
        raise NotImplementedError

    return model_out


def reload_ncsn(model, config, task_name, ds_name):
    if "Diffusion" in task_name:
        task_name = "Diffusion"
    mode = _get_data_mode(config)
    ds_dict = {}
    params = {
        "batch_size": config.training.batch_size,
        "lr": config.optim.lr,
        "data_mode": mode
    }
    lit_model = TrainScoreModelDiscrete(model, ds_dict, params)

    ckpt_path_pattern = os.path.join(RELOAD_ROOT_DIRS[task_name], RELOAD_MODEL_DIRS[task_name][ds_name][mode],
                                     "checkpoints", "*.ckpt")
    ckpt_path = glob.glob(ckpt_path_pattern)[0]
    ckpt = torch.load(ckpt_path)
    lit_model.load_from_checkpoint(ckpt_path, score_model=model, ds_dict=ds_dict, params=params, map_location=ptu.DEVICE)
    state_dict_out = collate_state_dict(ckpt["callbacks"]["EMA"]["ema_state_dict"])
    model_reload = lit_model.model.to(ptu.DEVICE)
    model_reload.load_state_dict(state_dict_out)
    model_reload.eval()

    return model_reload


def reload_clf(model, config, task_name, ds_name):
    mode = _get_data_mode(config)
    ds_dict = {}
    params = {
        "batch_size": config.training.batch_size,
        "lr": config.optim.lr,
        "data_mode": mode
    }
    lit_model = TrainClf(model, ds_dict, params, config)
    ckpt_path_pattern = os.path.join(RELOAD_ROOT_DIRS[task_name], RELOAD_MODEL_DIRS[task_name][ds_name][mode],
                                     "checkpoints", "*.ckpt")
    ckpt_path = glob.glob(ckpt_path_pattern)[0]
    lit_model.load_from_checkpoint(ckpt_path, model=model, ds_dict=ds_dict, params=params, config=config, map_location=ptu.DEVICE)
    model_reload = lit_model.model.to(ptu.DEVICE)
    model_reload.eval()

    return model_reload


def reload_seg(model, config, task_name, ds_name):
    mode = _get_data_mode(config)
    ds_dict = {}
    params = {
        "batch_size": config.training.batch_size,
        "lr": config.optim.lr,
        "data_mode": mode,
        "num_cls": 2
    }
    lit_model = TrainSeg(model, ds_dict, params, config)
    ckpt_path_pattern = os.path.join(RELOAD_ROOT_DIRS[task_name], RELOAD_MODEL_DIRS[task_name][ds_name][mode],
                                     "checkpoints", "*.ckpt")
    ckpt_path = glob.glob(ckpt_path_pattern)[0]
    lit_model.load_from_checkpoint(ckpt_path, model=model, ds_dict=ds_dict, params=params, config=config, map_location=ptu.DEVICE)
    model_reload = lit_model.model.to(ptu.DEVICE)
    model_reload.eval()

    return model_reload


def _get_data_mode(config):
    mode = "real-valued"
    # if config.data.channels == 1:
    #     mode = "real-valued"
    if config.data.channels == 2:
        mode = "complex"
    # else:
    #     raise ValueError("Invalid number of channels.")

    return mode
