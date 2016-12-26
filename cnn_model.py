#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
from chainer import computational_graph
import chainer.functions as F
import chainer.links as L
import math
from chainer import report, training, Chain, datasets, iterators, optimizers
import six
import time
import numpy as np
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import argparse
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy
import sys
import pickle
import matplotlib.pyplot as plt
from chainer import serializers


# CGP(list)からCNNのモデル構築
# n_in:  入力画像のchannel
# n_out: 畳込み層の出力channel数
# Classifierで定義想定（ model = Classifier(CGP2CNN(cgp)) )
class CGP2CNN(chainer.Chain):
    def __init__(self, cgp, n_in=3, n_out=32):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        links = []
        i = 1
        for name, in1, in2 in self.cgp:
            if name == 'conv3':
                links += [(name+'_'+str(i), L.Convolution2D(None, n_out, 3, pad=1))]
            elif name == 'conv5':
                links += [(name+'_'+str(i), L.Convolution2D(None, n_out, 5, pad=2))]
            elif name == 'conv7':
                links += [(name+'_'+str(i), L.Convolution2D(None, n_out, 7, pad=3))]
            elif name == 'batch':
                if in1 == 0:
                    links += [(name+'_'+str(i), L.BatchNormalization(n_in))]
                else:
                    links += [(name+'_'+str(i), L.BatchNormalization(n_out))]                        
            elif name == 'pool_max':
                links += [('_'+name+'_'+str(i), F.MaxPooling2D(self.pool_size, self.pool_size, 0, False))]
            elif name == 'pool_ave':
                links += [('_'+name+'_'+str(i), F.AveragePooling2D(self.pool_size, self.pool_size, 0, False))]
            elif name == 'ReLU':
                links += [('_'+name+'_'+str(i), F.ReLU())]
            elif name == 'tanh':
                links += [('_'+name+'_'+str(i), F.Tanh())]
            elif name == 'concat':
                links += [('_'+name+'_'+str(i), F.Concat())]
            elif name == 'full':
                links += [(name+'_'+str(i), L.Linear(None, 10))]
            i += 1
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
        self.train = True
        self.lossfun = softmax_cross_entropy
        self.accuracy = None
        self.accfun = accuracy

    def __call__(self, x):
        outputs = []
        for i in range(len(self.forward)+1):
            outputs.append(x) # 原画像
        nodeID = 1        
        for name, f in self.forward:
            if 'conv' in name:
                x = getattr(self, name)(outputs[self.cgp[nodeID][1]])
                outputs[nodeID] = x
                nodeID += 1
            elif name.startswith('_') and not 'concat' in name:
                x = f(outputs[self.cgp[nodeID][1]])
                outputs[nodeID] = x
                nodeID += 1
            elif 'batch' in name:
                x = getattr(self, name)(outputs[self.cgp[nodeID][1]], not self.train)
                outputs[nodeID] = x
                nodeID += 1
            elif 'concat' in name:
                if outputs[self.cgp[nodeID][1]].shape[2] != outputs[self.cgp[nodeID][2]].shape[2]:
                    ratio = outputs[self.cgp[nodeID][1]].shape[2] / outputs[self.cgp[nodeID][2]].shape[2]
                    if ratio < 1:
                        ratio = outputs[self.cgp[nodeID][2]].shape[2] / outputs[self.cgp[nodeID][1]].shape[2]
                        ratio = ratio / 2
                        ratio = int(ratio)
                        y = outputs[self.cgp[nodeID][2]]
                        for p in range(ratio):
                            y = F.max_pooling_2d(y, self.pool_size, self.pool_size, 0, False)
                        x = f(outputs[self.cgp[nodeID][1]], y)
                    else:
                        ratio = ratio / 2
                        ratio = int(ratio)
                        y = outputs[self.cgp[nodeID][1]]
                        for p in range(ratio):
                            y = F.max_pooling_2d(y, self.pool_size, self.pool_size, 0, False)
                        x = f(y, outputs[self.cgp[nodeID][2]])
                else:
                    x = f(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]])
                outputs[nodeID] = x
                nodeID += 1
            else:
                x = getattr(self, name)(outputs[self.cgp[nodeID][1]])
                outputs[nodeID] = x
                nodeID += 1
        if self.train:    
            # self.loss = F.softmax_cross_entropy(x, t)
            # self.accuracy = F.accuracy(x, t)
            # return self.loss
            return x            
        else:
            return x