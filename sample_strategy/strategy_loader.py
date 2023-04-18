import numpy as np
import torch
from .RandomSample import RandomSampling
from .STGADA import STGADA

# load different sample strategy
def load_strategy(strategy, source, target_train, target_test, idxs_lb, net, cfg, logger, target_train_fda=None):
    if strategy.lower() == 'random':
        # random
        return RandomSampling(source, target_train, target_test, idxs_lb, net, cfg)
    elif strategy.lower() == 'stgada':
        # sdm
        return STGADA(source, target_train, target_test, idxs_lb, net, cfg, logger, target_train_fda)
    else:
        raise ValueError
