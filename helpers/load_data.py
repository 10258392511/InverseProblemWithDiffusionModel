import os

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize


def load_mnist(root_dir, mode="train"):
    assert mode in ["train", "val", "test"]

    transforms = [
        ToTensor(),
        Normalize(mean=[0.], std=[1.]),
    ]
    transforms = Compose(transforms)
    if_train = True if mode == "train" else False
    ds = MNIST(root_dir, train=if_train, transform=transforms, download=True)

    return ds


def load_cifar10(root_dir, mode="train"):
    assert mode in ["train", "val", "test"]
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    transforms = [
        ToTensor(),
        Normalize(mean=[0.], std=[1.]),
    ]
    transforms = Compose(transforms)
    if_train = True if mode == "train" else False
    ds = CIFAR10(root_dir, train=if_train, transform=transforms, download=True)

    return ds
