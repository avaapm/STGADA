import os
import torch
import numpy as np
import math
import random
from torch.utils.data import DataLoader

import torch.optim as optim
from sklearn.mixture import GaussianMixture

import config.load_config as cfg
import data.dataset as loader
import data.transform as transform
from util.utils import init_label_pool, setLogger
from model.resnet import ResNet50Fc
from sample_strategy.strategy_loader import load_strategy


def main():
    # set random seed
    global device;device = torch.device("cuda:" + cfg.DEVICE if cfg.USE_CUDA else "cpu")
    kwargs = {'num_workers': cfg.NUM_WORK, 'pin_memory': True} if cfg.USE_CUDA else {}
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    currentDir = os.path.dirname(os.path.realpath(__file__))
    result_dir = os.path.join(currentDir, cfg.DATA_NAME + '_result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    logFile = os.path.join(result_dir, '{}-{}.log'.format(cfg.DATA_SOURCE, cfg.DATA_TARGET))
    logger = setLogger(logFile)


    # load train_source_data train_target_data and test_target_data
    train_source_data = loader.get_data(cfg.DATA_NAME,os.path.join(cfg.DATA_PATH, cfg.DATA_SOURCE),
    transform.train_transform,tr_or_te='train',n_views=cfg.N_VIEWS)
    train_target_data = loader.get_data(cfg.DATA_NAME,os.path.join(cfg.DATA_PATH, cfg.DATA_TARGET),
    transform.train_transform,tr_or_te='train',n_views=cfg.N_VIEWS)
    test_target_data = loader.get_data(cfg.DATA_NAME,os.path.join(cfg.DATA_PATH, cfg.DATA_TARGET_TEST),
    transform.test_transform,tr_or_te='test',n_views=cfg.N_VIEWS)

    train_target_data_fda = loader.get_data('RGB_fda',os.path.join(cfg.DATA_PATH),
    transform.train_transform,tr_or_te='train',n_views=cfg.N_VIEWS)

    source_train_loader = DataLoader(train_source_data, batch_size=cfg.BATCH_SIZE, 
    shuffle=True, drop_last=True,**kwargs)
    target_train_loader = DataLoader(train_target_data, batch_size=cfg.BATCH_SIZE,drop_last=True,**kwargs)
    target_test_loader = DataLoader(test_target_data, batch_size=cfg.BATCH_SIZE,drop_last=True,**kwargs)

    target_train_loader_fda = DataLoader(train_target_data_fda, batch_size=cfg.BATCH_SIZE, drop_last=True, **kwargs)

    # init label pool
    n_pool = len(train_target_data)
    idxs_lb = init_label_pool(n_pool,cfg.NUM_INIT_LB)
    num_active = math.ceil(n_pool * cfg.QUERY_RATIO)

    # load model
    net = ResNet50Fc(class_num = cfg.DATA_CLASS)



    # select strategy
    strategy = load_strategy(cfg.SAMPLE_STRATEGY, 
    source_train_loader, target_train_loader, target_test_loader, idxs_lb, net, cfg, logger, target_train_loader_fda)

    logger.info('-----------------------------------------------------------')
    logger.info('Start Sample Strategy %s with data %s --> %s'%(type(strategy).__name__,cfg.DATA_SOURCE,cfg.DATA_TARGET))
    logger.info('-----------------------------------------------------------')
    best_acc, best_model = 0.0, None
    for epoch in range(1,cfg.EPOCH+1):
        strategy.train_STGADA(epoch)
        if epoch in [11, 13, 15, 17, 19]:
            active_model = strategy.clf.state_dict()
            torch.save(active_model, os.path.join(result_dir, "active_model_{}_{}_{}.pth".format(epoch, cfg.DATA_SOURCE, cfg.DATA_TARGET)))
        if epoch in [10, 12, 14, 16, 18]:
            # query samples with different active learning strategy
            query_indx = strategy.STGADA_query(num_active)
            strategy.sdm_active(query_indx, train_target_data, train_source_data)

        avgAcc = strategy.test()
        if avgAcc > best_acc:
            best_model = strategy.clf.state_dict()
            best_acc = avgAcc
            torch.save(best_model, os.path.join(result_dir, "best_model_{}_{}.pth".format(cfg.DATA_SOURCE, cfg.DATA_TARGET)))
    
    
    return best_acc



if __name__ == "__main__":
    acc = main()

