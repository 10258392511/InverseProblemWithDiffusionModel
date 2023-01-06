import abc


class AbstractRegularizer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ZeroReg(AbstractRegularizer):
    def __call__(self, *args, **kwargs):
        return 0
