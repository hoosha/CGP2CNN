#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy
from chainer import reporter

# CONV → Batch → ReLU を1つのノードとして返すクラス
class ConvBlock(chainer.Chain):
    def __init__(self, ksize, initializer, n_out=32):
        super(ConvBlock, self).__init__()
        links = []
        if ksize == 3:
            links = [('conv1', L.Convolution2D(None, n_out, ksize, pad=1, initialW=initializer))]
            links += [('bn1', L.BatchNormalization(n_out))]
        elif ksize == 5:
            links = [('conv1', L.Convolution2D(None, n_out, ksize, pad=2, initialW=initializer))]
            links += [('bn1', L.BatchNormalization(n_out))]
        else:
            links = [('conv1', L.Convolution2D(None, n_out, 7, pad=3, initialW=initializer))]
            links += [('bn1', L.BatchNormalization(n_out))]
        
        for link in links:
            self.add_link(*link)
        self.forward = links
    
    def __call__(self, x, train):
        for name, f in self.forward:
            if 'conv1' in name:
                x = getattr(self, name)(x)
            elif 'bn1' in name:
                x = getattr(self, name)(x, not train)
        return F.relu(x)

# Batch → ReLU を1つのノードとして返すクラス（単独でReLUノードを使わない場合，不必要）
class ActivationBlock(chainer.Chain):
    def __init__(self):
        super(ActivationBlock, self).__init__()
        links = []
        links = [('bn1', L.BatchNormalization(None))]            
        links += [('_act1', F.ReLU())]
        
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
    
    def __call__(self, x, train):
        for name, f in self.forward:
            if 'bn1' in name:
                x = getattr(self, name)(x, not train)
            elif name.startswith('_'):
                x = f(x)
        return x


# CGP(list)からCNNのモデル構築
# n_in:  入力画像のchannel
# n_out: 畳込み層の出力channel数
# Classifierで定義想定（ model = Classifier(CGP2CNN(cgp)) )
class CGP2CNN(chainer.Chain):
    def __init__(self, cgp, n_class, n_in=3, n_out=32, lossfun=softmax_cross_entropy.softmax_cross_entropy, accfun=accuracy.accuracy):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        self.n_out= n_out
        initializer = chainer.initializers.HeNormal()
        links = []
        i = 1
        for name, in1, in2 in self.cgp:
            if name == 'conv3':
                links += [(name+'_'+str(i), L.Convolution2D(None, n_out, 3, pad=1, initialW=initializer))]
            elif name == 'conv5':
                links += [(name+'_'+str(i), L.Convolution2D(None, n_out, 5, pad=2, initialW=initializer))]
            elif name == 'conv7':
                links += [(name+'_'+str(i), L.Convolution2D(None, n_out, 7, pad=3, initialW=initializer))]
            elif name == 'pool_max':
                links += [('_'+name+'_'+str(i), F.MaxPooling2D(self.pool_size, self.pool_size, 0, False))]
            elif name == 'pool_max_const':
                links += [('_'+name+'_'+str(i), F.MaxPooling2D(3, 1, 1, True))]
            elif name == 'pool_ave':
                links += [('_'+name+'_'+str(i), F.AveragePooling2D(self.pool_size, self.pool_size, 0, False))]
            elif name == 'pool_ave_const':
                links += [('_'+name+'_'+str(i), F.AveragePooling2D(3, 1, 1, True))]
            elif name == 'ReLU':
                links += [('ActivationBlock'+'_'+str(i), ActivationBlock())]  # ◆BatchNormalization(None)にしちゃうとBatchNormalizationのパラメータを学習してくれない?（精度が悪くなる）
                # links += [('batch_'+str(i), L.BatchNormalization(None))]
                # links += [('_'+name+'_'+str(i), F.ReLU())]
            elif name == 'tanh':
                links += [('batch_'+str(i), L.BatchNormalization(None))]
                links += [('_'+name+'_'+str(i), F.Tanh())]
            elif name == 'concat':
                links += [('_'+name+'_'+str(i), F.Concat())]
            elif name == 'sum':
                links += [('_'+name+'_'+str(i), F.Concat())]
            elif name == 'ConvBlock3':
                links += [(name+'_'+str(i), ConvBlock(3, initializer))]
            elif name == 'ConvBlock5':
                links += [(name+'_'+str(i), ConvBlock(5, initializer))]
            elif name == 'ConvBlock7':
                links += [(name+'_'+str(i), ConvBlock(7, initializer))]
            elif name == 'full':
                links += [(name+'_'+str(i), L.Linear(None, n_class, initialW=initializer))]
            i += 1
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
        self.train = True
        self.lossfun = lossfun
        self.accfun = accfun
        self.loss = None
        self.accuracy = None
        self.outputs = [None for _ in range(len(self.cgp))]
        self.param_num = 0

    def __call__(self, x, t):
        xp = chainer.cuda.get_array_module(x)
        outputs = self.outputs
        outputs[0] = x    # 原画像
        nodeID = 1
        param_num = 0
        for name, f in self.forward:
            if 'conv' in name:
                outputs[nodeID] = getattr(self, name)(outputs[self.cgp[nodeID][1]])
                nodeID += 1
                param_num += (f.W.shape[0]*f.W.shape[2]*f.W.shape[3]*f.W.shape[1]+f.W.shape[0])
            elif 'ConvBlock' in name:
                outputs[nodeID] = getattr(self, name)(outputs[self.cgp[nodeID][1]], self.train)
                nodeID += 1
            elif 'ActivationBlock' in name:
                outputs[nodeID] = getattr(self, name)(outputs[self.cgp[nodeID][1]], self.train)
                nodeID += 1
            # elif 'batch' in name:
            #     outputs[nodeID] = getattr(self, name)(outputs[self.cgp[nodeID][1]], not self.train)
            #     param_num += outputs[nodeID].data.shape[1]*2
            # elif 'ReLU' in name or 'tanh' in name:
            #     outputs[nodeID] = f(outputs[nodeID])
            #     nodeID += 1
            elif name.startswith('_') and 'concat' not in name and 'sum' not in name:
                outputs[nodeID] = f(outputs[self.cgp[nodeID][1]])
                nodeID += 1
            elif 'concat' in name:
                in_data = [outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]]]
                small_in_id, large_in_id = (0, 1) if in_data[0].shape[2] < in_data[1].shape[2] else (1, 0)
                pool_num = xp.floor(xp.log2(in_data[large_in_id].shape[2] / in_data[small_in_id].shape[2]))
                for _ in xp.arange(pool_num):
                    in_data[large_in_id] = F.max_pooling_2d(in_data[large_in_id], self.pool_size, self.pool_size, 0, False)
                outputs[nodeID] = f(in_data[0], in_data[1])
                nodeID += 1
            elif 'sum' in name:
                in_data = [outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]]]
                # 画像サイズに関するチェック
                small_in_id, large_in_id = (0, 1) if in_data[0].shape[2] < in_data[1].shape[2] else (1, 0)
                pool_num = xp.floor(xp.log2(in_data[large_in_id].shape[2] / in_data[small_in_id].shape[2]))
                for _ in xp.arange(pool_num):
                    in_data[large_in_id] = F.max_pooling_2d(in_data[large_in_id], self.pool_size, self.pool_size, 0, False)
                # channel sizeに関するチェック
                small_ch_id, large_ch_id = (0, 1) if in_data[0].shape[1] < in_data[1].shape[1] else (1, 0)
                pad_num = int(in_data[large_ch_id].shape[1] - in_data[small_ch_id].shape[1])
                tmp = in_data[large_ch_id][:, :pad_num, :, :]
                in_data[small_ch_id] = F.concat((in_data[small_ch_id], tmp * 0), axis=1)
                outputs[nodeID] = in_data[0] + in_data[1]
                nodeID += 1
            else:
                outputs[nodeID] = getattr(self, name)(outputs[self.cgp[nodeID][1]])
                nodeID += 1
                param_num += f.W.data.shape[0] * f.W.data.shape[1] + f.b.data.shape[0]
        self.param_num = param_num

        if self.train:    
            # self.loss = F.softmax_cross_entropy(outputs[-1], t)
            # self.accuracy = F.accuracy(outputs[-1], t)
            # return self.loss

            self.loss = None
            self.accuracy = None
            self.loss = self.lossfun(outputs[-1], t)
            reporter.report({'loss': self.loss}, self)
            self.accuracy = self.accfun(outputs[-1], t)
            reporter.report({'accuracy': self.accuracy}, self)
            return self.loss

            # return outputs[-1]
        else:
            return outputs[-1]
