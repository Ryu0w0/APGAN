from abc import ABCMeta, abstractmethod, abstractclassmethod
import numpy as np
from torch import optim
from torch import nn
from common.logger import logger_


class AbstractSubAPGAN(nn.Module, metaclass=ABCMeta):
    """
    It can be inherited by discriminator and generator as a sub-network of GAN
    """
    def __init__(self, config, device):
        super().__init__()
        # variables
        self.config = config
        self.blocks = self._build_block()
        self.rgb_layers = self._build_rgb_layer()
        self.optimizer = self.get_optimizer()
        self.initialized_layers = dict()
        self.device = device
        self.alpha = 0

        # module weight initialization
        self._initialization()

    def set_alpha(self, alpha):
        self.alpha = alpha

    @staticmethod
    @abstractmethod
    def _build_rgb_layer():
        pass

    @abstractmethod
    def _build_block(self):
        pass

    def _initialization(self):
        self.apply(self._initialize_weight)
        for k, v in self.initialized_layers.items():
            logger_.info(f"Num of initialized {k}: {v}")

    @abstractmethod
    def _initialize_weight(self, m):
        pass

    def _store_init_layer_nm(self, cls_name):
        if cls_name in self.initialized_layers.keys():
            self.initialized_layers[cls_name] += 1
        else:
            self.initialized_layers[cls_name] = 1

    def get_optimizer(self):
        config = self.config["opt"]
        if config["type"] == "adam":
            lr = config["lr"]
            beta1, beta2 = config["betas"]
            optimizer = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-8)
            return optimizer
        else:
            assert False, f"Unexpected config: {config}"

    @classmethod
    @abstractmethod
    def check_structure(cls, config, max_step_idx, device):
        pass

