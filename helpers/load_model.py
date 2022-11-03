from InverseProblemWithDiffusionModel.ncsn.models.ncsnv2 import NCSNv2Deepest
from InverseProblemWithDiffusionModel.ncsn.models.classifiers import ResNetClf
from monai.networks.nets import UNet
from InverseProblemWithDiffusionModel.helpers.load_data import load_config


TASK_NAME_TO_MODEL_CTOR = {
    "Diffusion": NCSNv2Deepest,
    "Clf": ResNetClf,
    "Seg": UNet
}


def load_model(config, task_name):
    pass
