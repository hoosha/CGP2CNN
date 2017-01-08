#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pickle
import pandas as pd

from cgp import *
from cgp_config import *


if __name__ == '__main__':
    network_info = CgpInfoCnn()
    with open('network_info.pickle', mode='wb') as f:
        pickle.dump(network_info, f)

    eval_f = ThreadingCNNEvaluation(gpu_num=2, dataset='cifar10', valid_data_ratio=0.1, verbose=True, epoch_num=50, batchsize=256)
    cgp = ModifyCGP(network_info, eval_f, gpu_num=2)
    cgp.evolution(max_eval=2000, mutation_rate=0.1, log_file='./log_cgp.txt')

    # restart
    '''
    print('Restart!!')
    with open('network_info.pickle', mode='rb') as f:
        network_info = pickle.load(f)
    cgp = CGP(network_info, TestFunc(), lam=4)

    data = pd.read_csv('./log.txt', header=None)
    cgp.load_log(list(data.tail(1).values.flatten().astype(int)))
    cgp.evolution(max_eval=20, mutation_rate=0.05, log_file='./log_restat.txt')
    '''