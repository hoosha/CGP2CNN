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
from sklearn.datasets import fetch_mldata
#自作クラスのインポート
from cnn_model import CGP2CNN

# cifar10のロードに使用
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')                                                                                    
    fp.close()
    return data

# __init__: datasetのロード
# __call__: cgp(list)からCNNの構築，CNNの学習
# valid_data_ratio: 全学習データに対するvalidation data の割合(e.g. ratio=0.2 の場合，train=40000, valid=10000 )
class CNN_train():
    def __init__(self, dataset_name, valid_data_ratio=0.2):
        data = None
        self.x_train = None
        self.x_valid = None
        label = []
        self.y_train = None
        self.y_valid = None
        self.train_data_num = None
        self.valid_data_num = None
        self.test_data_num  = None
        # load dataset
        if dataset_name == 'cifar10':
            for i in range(1,6):
                data_dic = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
                if i == 1:
                    data = data_dic['data']
                else:
                    data = np.vstack((data, data_dic['data']))
                label += data_dic['labels']
            test_data_dic = unpickle("cifar-10-batches-py/test_batch")
            self.x_test = test_data_dic['data']
            self.x_test = self.x_test.reshape(len(self.x_test),3, 32, 32)
            self.y_test = np.array(test_data_dic['labels'])
            N = len(data)
            N = int(N * valid_data_ratio)
            self.x_valid, self.x_train = np.split(data, [N])
            self.x_train = self.x_train.reshape(len(self.x_train),3, 32, 32)
            self.x_valid = self.x_valid.reshape(len(self.x_valid),3, 32, 32)
            self.y_valid, self.y_train = np.split(label, [N])
            self.x_train = self.x_train.astype(np.float32)
            self.x_valid = self.x_valid.astype(np.float32)
            self.x_test = self.x_test.astype(np.float32)
            self.x_train /= 255
            self.x_valid /= 255
            self.x_test /= 255
            label = np.array(label)                                                                         
            self.y_train = self.y_train.astype(np.int32)
            self.y_valid = self.y_valid.astype(np.int32)
            self.y_test = self.y_test.astype(np.int32)
            self.train_data_num = len(self.x_train)
            self.valid_data_num = len(self.x_valid)
            self.test_data_num = len(self.x_test)
            print('data shape:', self.x_train.shape)
            print('train data num:', self.train_data_num)
            print('valid data num:', self.valid_data_num)
            print('test  data num:', self.test_data_num)
        elif dataset_name == 'mnist':
            mnist = fetch_mldata('MNIST original')
            mnist.data   = mnist.data.astype(np.float32)
            mnist.data  /= 255     # 0-1のデータに変換
            mnist.target = mnist.target.astype(np.int32)
            N = 60000
            data, self.x_test = np.split(mnist.data,   [N])
            label, self.y_test = np.split(mnist.target, [N])
            N = len(data)
            N = int(N * valid_data_ratio)
            self.x_valid, self.x_train = np.split(data, [N])
            self.y_valid, self.y_train = np.split(label, [N])
            self.x_train = self.x_train.reshape(len(self.x_train), 1, 28, 28)
            self.x_valid = self.x_valid.reshape(len(self.x_valid), 1, 28, 28)
            self.x_test = self.x_test.reshape(len(self.x_test), 1, 28, 28)
            self.train_data_num = len(self.x_train)
            self.valid_data_num = len(self.x_valid)
            self.test_data_num = len(self.x_test)
            print('data shape:', self.x_train.shape)
            print('train data num  :', self.train_data_num)
            print('valid data num  :', self.valid_data_num)
            print('test data num   :', self.test_data_num)
        else:
            print('input dataset_name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, epoch=200, batchsize=256):
        print('GPUID    :', gpuID)
        print('epoch    :', epoch)
        print('batchsize:', batchsize)
        model = L.Classifier(CGP2CNN(cgp))
        chainer.cuda.get_device(gpuID).use()  # Make a specified GPU current
        model.to_gpu(gpuID)                   # Copy the model to the GPU
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        for epoch in six.moves.range(1, epoch+1):
            print('epoch', epoch)
            perm = np.random.permutation(self.train_data_num)
            sum_accuracy = 0
            sum_loss = 0
            start = time.time()
            for i in six.moves.range(0, self.train_data_num, batchsize):
                x = chainer.Variable(cuda.to_gpu(self.x_train[perm[i:i + batchsize]]))
                t = chainer.Variable(cuda.to_gpu(self.y_train[perm[i:i + batchsize]]))
                optimizer.update(model, x, t)

                if epoch == 1 and i == 0:
                    with open('graph.dot', 'w') as o:
                        g = computational_graph.build_computational_graph((model.loss, ))
                        o.write(g.dump())
                    print('CNN graph generated')

                sum_loss += float(model.loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)
            end = time.time()
            elapsed_time = end - start
            throughput = self.train_data_num / elapsed_time
            print('train mean loss={}, train accuracy={}, time={}, throughput={} images/sec'.format(sum_loss / self.train_data_num, sum_accuracy / self.train_data_num, elapsed_time, throughput))
            # 検証画像による検証
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, self.valid_data_num, batchsize):
                x = chainer.Variable(cuda.to_gpu(self.x_valid[i:i + batchsize]))
                t = chainer.Variable(cuda.to_gpu(self.y_valid[i:i + batchsize]))
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)
            print('valid  mean loss={}, valid accuracy={}'.format(sum_loss / self.valid_data_num, sum_accuracy / self.valid_data_num))
        # モデルの保存
        model.to_cpu()
        serializers.save_npz("mymodel.model", model)
        return sum_accuracy / self.valid_data_num




# 確認用
# cgp = []
# cgp.append(['Input',0,0])     #0
# cgp.append(['conv3',0,0])     #1
# cgp.append(['batch',1,0])     #2
# cgp.append(['ReLU',2,0])      #3
# cgp.append(['conv3',3,0])     #4
# cgp.append(['conv5',3,0])     #5
# cgp.append(['conv7',3,4])     #6
# cgp.append(['pool_max',6,0])  #7
# cgp.append(['batch',4,0])     #8
# cgp.append(['batch',5,0])     #9 
# cgp.append(['batch',7,0])     #10
# cgp.append(['ReLU',10,0])     #11
# cgp.append(['ReLU',8,0])      #12
# cgp.append(['ReLU',9,0])      #13
# cgp.append(['concat',12,13])  #14
# cgp.append(['sum',14,11])     #15
# cgp.append(['full',15,12])    #16

cgp = []
cgp.append(['Input',0,0])     #0
cgp.append(['conv3',0,0])     #1
cgp.append(['ReLU',1,0])     #2
cgp.append(['ReLU',2,0])      #3
cgp.append(['conv3',3,0])     #4
cgp.append(['conv5',3,0])     #5
cgp.append(['conv7',3,4])     #6
cgp.append(['pool_max',6,0])  #7
cgp.append(['ReLU',4,0])     #8
cgp.append(['ReLU',5,0])     #9 
cgp.append(['ReLU',7,0])     #10
cgp.append(['concat',8,9])   #11
cgp.append(['concat',11,10])     #12
cgp.append(['full',12,11])    #13

temp = CNN_train('cifar10')
acc = temp(cgp, 1)
