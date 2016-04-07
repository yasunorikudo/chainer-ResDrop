#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

count = None

class BottleNeckA(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(out_size),

            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True),
            bn4=L.BatchNormalization(out_size),
        )

    def __call__(self, x, train, decay):
        global count
        count += 1
        if decay[count][0] is 1:
            h1 = F.relu(self.bn1(self.conv1(x), test=not train))
            h1 = F.relu(self.bn2(self.conv2(h1), test=not train))
            h1 = self.bn3(self.conv3(h1), test=not train)
            h2 = self.bn4(self.conv4(x), test=not train)
            return F.relu(h1 + h2) if train else F.relu(h1 * decay[count][1] + h2)
        else:
            return F.relu(self.bn4(self.conv4(x), test=not train))


class BottleNeckB(chainer.Chain):
    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(in_size),
        )

    def __call__(self, x, train, decay):
        global count
        count += 1
        if decay[count][0] is 1:
            h = F.relu(self.bn1(self.conv1(x), test=not train))
            h = F.relu(self.bn2(self.conv2(h), test=not train))
            h = self.bn3(self.conv3(h), test=not train)
            return F.relu(h + x) if train else F.relu(h * decay[count][1] + x)
        else:
            return x


class Block(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB(out_size, ch))]

        for link in links:
            self.add_link(*link)
        self.forward = links

    def __call__(self, x, train, decay):
        for name,_ in self.forward:
            f = getattr(self, name)
            h = f(x if name == 'a' else h, train, decay)

        return h


class ResNet(chainer.Chain):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        self.R = (3, 4, 6, 3)
        super(ResNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(self.R[0], 64, 64, 256, 1),
            res3=Block(self.R[1], 256, 128, 512),
            res4=Block(self.R[2], 512, 256, 1024),
            res5=Block(self.R[3], 1024, 512, 2048),
            fc=L.Linear(2048, 1000),
        )
        self.train = True

    def clear(self):
        global count
        count = -1
        self.loss = None
        self.accuracy = None

    def resdrop(self, pL=0.5):
        L = sum(self.R)
        arr = []
        for l in range(1, L+1):
            pl = 1 - l * (1 - pL) / L
            arr.append([np.random.binomial(1, pl) if self.train else 1, pl])
        self.decay = arr

    def __call__(self, x, t):
        self.clear()
        self.resdrop()
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train, self.decay)
        h = self.res3(h, self.train, self.decay)
        h = self.res4(h, self.train, self.decay)
        h = self.res5(h, self.train, self.decay)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            return self.loss
        else:
            return h
