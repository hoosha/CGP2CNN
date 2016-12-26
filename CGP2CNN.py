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


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
parser.add_argument('--model', '-m', default=0, help ='Trained Model')
args = parser.parse_args()


# CGPリストからCNNのモデル構築
class CGP2CNN(chainer.Chain):
    def __init__(self, cgp, n_in=1, n_out=32):
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
                


cgp = []
cgp.append(['Input',0,0])     #0
cgp.append(['conv3',0,0])     #1
cgp.append(['batch',1,0])     #2
cgp.append(['ReLU',2,0])      #3
cgp.append(['conv3',3,0])     #4
cgp.append(['conv5',3,0])     #5
cgp.append(['conv7',3,4])     #6
cgp.append(['pool_max',6,0])  #7
cgp.append(['batch',4,0])     #8
cgp.append(['batch',5,0])     #9 
cgp.append(['batch',7,0])     #10
cgp.append(['ReLU',10,0])     #11
cgp.append(['ReLU',8,0])      #12
cgp.append(['ReLU',9,0])      #13
cgp.append(['concat',12,13])  #14
cgp.append(['concat',14,11])  #15
cgp.append(['full',15,12])    #16

batchsize = 128
epoch = 5

if args.model != 0:
    model = L.Classifier(CGP2CNN(cgp))
    serializers.load_npz(args.model, model)
else:
    model = L.Classifier(CGP2CNN(cgp))

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU
# Setup an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# # Load the MNIST dataset
# train, test = chainer.datasets.get_mnist(ndim=3)
# train_iter = chainer.iterators.SerialIterator(train, batchsize)
# test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

# Load the CIFAR10 dataset
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')                                                                                    
    fp.close()

    return data

x_train = None
y_train = []
for i in range(1,6):
    data_dic = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
    if i == 1:
        x_train = data_dic['data']
    else:
        x_train = np.vstack((x_train, data_dic['data']))
    y_train += data_dic['labels']

test_data_dic = unpickle("cifar-10-batches-py/test_batch")
x_test = test_data_dic['data']
x_test = x_test.reshape(len(x_test),3, 32, 32)
y_test = np.array(test_data_dic['labels'])
x_train = x_train.reshape((len(x_train),3, 32, 32))
y_train = np.array(y_train)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255                                                                                                                     
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
train = tuple_dataset.TupleDataset(x_train, y_train)
test = tuple_dataset.TupleDataset(x_test, y_test)
print('train data shape:', x_train.shape)
print('train data num  :', len(x_train))

# 画像の表示
n = 20
plt.figure(figsize=(20, 4))
nclasses = 10
pos = 1
for i in range(nclasses):
    # クラスiの画像のインデックスリストを取得
    targets = np.where(y_train == i)[0]
    np.random.shuffle(targets)
    # 最初の10枚の画像を描画
    for idx in targets[:10]:
        plt.subplot(10, 10, pos)
        img = x_train[idx]
        # (channel, row, column) => (row, column, channel)
        plt.imshow(img.reshape(3, 32, 32).transpose(1, 2, 0))
        plt.axis('off')
        pos += 1
plt.show()

train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

# Set up a trainer
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (epoch, 'epoch'), out=args.out)

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))
# Take a snapshot at each epoch
trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport(log_name='mylog'))

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

if args.resume:
    # Resume from a snapshot
    chainer.serializers.load_npz(args.resume, trainer)

# Run the training
trainer.run()

# モデルの保存
model.to_cpu()
serializers.save_npz("mymodel.model", model)
serializers.save_npz("mytrainer.state", optimizer)