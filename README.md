Deep Networks with Stochastic Depth implementation by Chainer
========

Implementation by Chainer. Original paper is [Deep Networks with Stochastic Depth](http://arxiv.org/abs/1603.09382).

This repository includes network definition scripts only.

If you want to train ResDrop from scratch, see [chainer sample code](https://github.com/pfnet/chainer/tree/master/examples/imagenet). 

# Requirements

- [Chainer 1.5+](https://github.com/pfnet/chainer) (Neural network framework)


# Usage

In python script, write:

```
from ResDrop152 import ResNet
model = ResNet()
```


# Traning speed

About 25% faster per iteration than ResNet with no layer drop.

# Sample result

I trained ResNet101 with layer drop and ResNet101 with no layer drop for PASCAL VOC Action dataset.
ResNet with layer drop improved the accuracy of test results about 4%.

![](https://raw.githubusercontent.com/wiki/yasunorikudo/chainer-ResDrop/images/result.png)
