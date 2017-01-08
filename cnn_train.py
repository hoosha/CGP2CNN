#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
from chainer import computational_graph
import chainer.links as L
import six
import time
import numpy as np
from chainer import serializers
import gc

# 自作クラスのインポート
from cnn_model import CGP2CNN


# __init__: datasetのロード
# __call__: cgp(list)からCNNの構築，CNNの学習
class CNN_train():
    # validation: [True]  モデル検証モード（trainデータを分割して学習データとテストデータを作成，GPでの評価用）
    #             [False] テストモード（全trainデータを学習データにし，testをテストデータにして性能評価，モデル最終評価用）
    # valid_data_ratio: モデル検証モードの際の，学習データとテストデータの分割の割合
    #                   (e.g., 全学習データ数=60000, ratio=0.2 の場合，train=40000, test=10000)
    # verbose: 表示のフラグ
    def __init__(self, dataset_name, validation=True, valid_data_ratio=0.1, verbose=True):
        self.verbose = verbose

        # load dataset
        if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'mnist':
            if dataset_name == 'cifar10':
                self.n_class = 10
                self.channel = 3
                train, test = chainer.datasets.get_cifar10(withlabel=True, ndim=3, scale=1.0)
            elif dataset_name == 'cifar100':
                self.n_class = 100
                self.channel = 3
                train, test = chainer.datasets.get_cifar100(withlabel=True, ndim=3, scale=1.0)
            else:    # mnist
                self.n_class = 10
                self.channel = 1
                train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3, scale=1.0)

            # モデル検証モード
            if validation:
                # split into train and validation data
                np.random.seed(2016)
                order = np.random.permutation(len(train))
                np.random.seed()
                if self.verbose:
                    print('data split order: ', order)
                train_size = int(len(train) * (1. - valid_data_ratio))

                # train data
                self.x_train, self.y_train = train[order[:train_size]][0], train[order[:train_size]][1]
                # test data
                self.x_test, self.y_test = train[order[train_size:]][0], train[order[train_size:]][1]
            # テストモード
            else:
                # train data
                self.x_train, self.y_train = train[range(len(train))][0], train[range(len(train))][1]
                # test data
                self.x_test, self.y_test = test[range(len(test))][0], test[range(len(test))][1]

        else:
            print('Invalid input dataset name at CNN_train()')
            exit(1)

        # data size
        self.train_data_num = len(self.x_train)
        self.test_data_num = len(self.x_test)
        if self.verbose:
            print('train data shape:', self.x_train.shape)
            print('test data shape :', self.x_test.shape)

    def __call__(self, cgp, gpuID, epoch_num=200, batchsize=256,
                 comp_graph='comp_graph.dot', out_model='mymodel.model', init_model=None):
        if self.verbose:
            print('\tGPUID    :', gpuID)
            print('\tepoch_num:', epoch_num)
            print('\tbatchsize:', batchsize)

        chainer.cuda.get_device(gpuID).use()  # Make a specified GPU current
        if init_model is None:
            # model = L.Classifier(CGP2CNN(cgp, self.n_class))
            model = CGP2CNN(cgp, self.n_class, n_in=self.channel) #パラメータ数を出すためこちらに変更
        else:
            if self.verbose:
                print('\tLoad model from', init_model)
            serializers.load_npz(init_model, model)
        model.to_gpu(gpuID)
        optimizer = chainer.optimizers.Adam()
        # optimizer = chainer.optimizers.NesterovAG()
        # optimizer = chainer.optimizers.RMSprop()
        # optimizer = chainer.optimizers.MomentumSGD()
        # optimizer = chainer.optimizers.AdaGrad()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

        for epoch in six.moves.range(1, epoch_num+1):
            if self.verbose:
                print('\tepoch', epoch)
            perm = np.random.permutation(self.train_data_num)
            train_accuracy = train_loss = 0
            start = time.time()
            for i in six.moves.range(0, self.train_data_num, batchsize):
                x = chainer.Variable(cuda.to_gpu(self.x_train[perm[i:i + batchsize]]))
                t = chainer.Variable(cuda.to_gpu(self.y_train[perm[i:i + batchsize]]))
                optimizer.update(model, x, t)

                if comp_graph is not None and epoch == 1 and i == 0:
                    with open(comp_graph, 'w') as o:
                        g = computational_graph.build_computational_graph((model.loss, ))
                        o.write(g.dump())
                        if self.verbose:
                            print('\tCNN graph generated.')

                train_loss += float(model.loss.data) * len(t.data)
                train_accuracy += float(model.accuracy.data) * len(t.data)
            elapsed_time = time.time() - start
            throughput = self.train_data_num / elapsed_time
            if self.verbose:
                print('\ttrain mean loss={}, train accuracy={}, time={}, throughput={} images/sec, paramNum={}'.format(train_loss / self.train_data_num, train_accuracy / self.train_data_num, elapsed_time, throughput, model.param_num))
            # テストデータへの適用
            if self.verbose:
                test_accuracy, test_loss = self.__test(model, batchsize)
                print('\tvalid  mean loss={}, valid accuracy={}'.format(test_loss / self.test_data_num, test_accuracy / self.test_data_num))

        test_accuracy, test_loss = self.__test(model, batchsize)

        model.to_cpu()
        if out_model is not None:
            serializers.save_npz(out_model, model)
        # del model
        # gc.collect()
        return test_accuracy / self.test_data_num

    def __test(self, model, batchsize):
        test_accuracy = test_loss = 0
        for i in six.moves.range(0, self.test_data_num, batchsize):
            x = chainer.Variable(cuda.to_gpu(self.x_test[i:i + batchsize]))
            t = chainer.Variable(cuda.to_gpu(self.y_test[i:i + batchsize]))
            loss = model(x, t)
            test_loss += float(loss.data) * len(t.data)
            test_accuracy += float(model.accuracy.data) * len(t.data)
        return test_accuracy, test_loss


if __name__ == '__main__':
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

    #cgp = []
    #cgp.append(['Input',0,0])     #0
    #cgp.append(['conv3',0,0])     #1
    #cgp.append(['ReLU',1,0])     #2
    #cgp.append(['ReLU',2,0])      #3
    #cgp.append(['conv3',3,0])     #4
    #cgp.append(['conv5',3,0])     #5
    #cgp.append(['conv7',3,4])     #6
    #cgp.append(['pool_max',6,0])  #7
    #cgp.append(['ReLU',4,0])     #8
    #cgp.append(['ReLU',5,0])     #9
    #cgp.append(['ReLU',7,0])     #10
    #cgp.append(['concat',8,9])   #11
    #cgp.append(['concat',11,10])     #12
    #cgp.append(['full',12,11])    #13

    cgp = [['input', 0, 0], ['ConvBlock3', 0, 0], ['ReLU', 1, 0], ['ConvBlock3', 0, 0], ['ConvBlock3', 2, 1], ['sum', 3, 4], ['full', 5, 6]]
    # cgp = [['input', 0, 0], ['ConvBlock3', 0, 0], ['pool_max', 0, 0], ['ConvBlock3', 1, 1], ['sum', 3, 2], ['full', 4, 6]]
    temp = CNN_train('cifar10')
    acc = temp(cgp, 0)
