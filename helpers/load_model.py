import yaml

from InverseProblemWithDiffusionModel.ncsn.models.ncsnv2 import NCSNv2Deepest
from InverseProblemWithDiffusionModel.ncsn.models.classifiers import ResNetClf
from monai.networks.nets import UNet
from InverseProblemWithDiffusionModel.helpers.load_data import load_config
from InverseProblemWithDiffusionModel.helpers.utils import load_yml_file

with open("../ncsn/configs/general_config.yml", "r") as rf:
    general_config = yaml.load(rf, yaml.Loader)


TASK_NAME_TO_MODEL_CTOR = {
    "Diffusion": NCSNv2Deepest,
    "Clf": ResNetClf,
    "Seg": UNet
}


def load_model(config, task_name):
    global general_config
    assert task_name in TASK_NAME_TO_MODEL_CTOR
    net_ctor = TASK_NAME_TO_MODEL_CTOR[task_name]
    if config is None:
        net_params = general_config[task_name]
        net_params["in_channels"] = config.data.channels

    model = None
    if task_name == "Diffusion":
        model = net_ctor(config)

    elif task_name == "Clf":
        model = net_ctor(net_params)

    elif task_name == "Seg":
        model = net_ctor(**net_params)

    return model
