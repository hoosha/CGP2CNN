#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import numpy as np
import cnn_train as cnn


class CnnEval(threading.Thread):
    def __init__(self, gpu_id):
        super(CnnEval, self).__init__()
        self.gpu_id = gpu_id
        self.model = None
        self.net = []
        self.epoch_num = 0
        self.batchsize = 0
        self.eval = 0

    def run(self):
        print('\tgpu_id:', self.gpu_id, ',', self.net)
        self.eval = self.model(self.net, self.gpu_id, epoch_num=self.epoch_num, batchsize=self.batchsize,
                               comp_graph='CNN.dot', out_model=None, init_model = None)
        print('\tgpu_id:', self.gpu_id, ', eval:', self.eval)


class ThreadingCNNEvaluation(object):
    def __init__(self, gpu_num, dataset='cifar10', valid_data_ratio=0.2, verbose=True, epoch_num=50, batchsize=256):
        self.model = [cnn.CNN_train(dataset, validation=True, valid_data_ratio=valid_data_ratio, verbose=verbose)
                      for _ in range(gpu_num)]
        self.gpu_num = gpu_num
        self.epoch_num = epoch_num
        self.batchsize = batchsize

    def __call__(self, net_lists):
        self.ths = [CnnEval(n) for n in range(self.gpu_num)]
        evaluations = np.zeros(len(net_lists))
        n = 0
        while n < len(net_lists):
            th_num = np.min((self.gpu_num, len(net_lists)-n))
            for i in range(th_num):
                self.ths[i].model = self.model[i]
                self.ths[i].net = net_lists[n + i]
                self.ths[i].epoch_num = self.epoch_num
                self.ths[i].batchsize = self.batchsize
                self.ths[i].start()
            for i in range(th_num):
                self.ths[i].join()
            for i in range(th_num):
                evaluations[n + i] = self.ths[i].eval
            n += th_num
        return evaluations


class CgpInfoCnn(object):
    def __init__(self, rows=30, cols=40):
        # network configurations depending on the problem
        self.input_num = 1

        self.func_type = ['conv3', 'conv5', 'conv7',
                          'ConvBlock3', 'ConvBlock5', 'ConvBlock7',
                          'pool_max',
                          'concat', 'sum']
        self.func_in_num = [1, 1, 1,
                            1, 1, 1,
                            1,
                            2, 2]

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.min_active_num = 8
        self.max_active_num = 50

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])


class TestEval(threading.Thread):
    def __init__(self):
        super(TestEval, self).__init__()
        self.g_truth = [['input', 0, 0],
                        ['conv3', 0, 0],
                        ['act.relu', 1, 0],
                        ['conv7', 0, 0],
                        ['pool.max', 2, 0],
                        ['concat', 3, 4],
                        ['full', 5, 0]
                        ]
        self.net = []
        self.eval = 0

    def run(self):
        # evaluating the simple difference the g_truth
        evaluation = 0
        for i, node in enumerate(self.g_truth):
            for j, c in enumerate(node):
                if len(self.net) > i and c == self.net[i][j]:
                    evaluation += 1
        self.eval = evaluation


class ThreadingTestEvaluation(object):
    def __init__(self, th_num):
        self.th_num = th_num

    def __call__(self, net_lists):
        self.ths = [TestEval() for _ in range(self.th_num)]
        evaluations = np.zeros(len(net_lists))
        n = 0
        while n < len(net_lists):
            th_num = np.min((self.th_num, len(net_lists)-n))
            for i in range(th_num):
                self.ths[i].net = net_lists[n + i]
                self.ths[i].start()
            for i in range(th_num):
                self.ths[i].join()
            for i in range(th_num):
                evaluations[n + i] = self.ths[i].eval
            n += th_num
        return evaluations


class CGPNetworkInfoTest(object):
    def __init__(self, rows=50, cols=30):
        # network configurations depending on the problem
        self.input_num = 1

        self.func_type = ['conv3', 'conv5', 'conv7', 'pool.max', 'act.relu', 'concat']
        self.func_in_num = [1, 1, 1, 1, 1, 2]

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.min_active_num = 8
        self.max_active_num = 50

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])
