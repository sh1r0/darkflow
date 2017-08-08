from .baseop import HEADER, LINE
from .convolution import conv_extract, conv_select, convolutional, local, reorg
from .simple import (avgpool, connected, crop, dropout, extract, flatten,
                     identity, leaky, maxpool, route, select, softmax)

op_types = {
    'convolutional': convolutional,
    'conv-select': conv_select,
    'connected': connected,
    'maxpool': maxpool,
    'leaky': leaky,
    'dropout': dropout,
    'flatten': flatten,
    'avgpool': avgpool,
    'softmax': softmax,
    'identity': identity,
    'crop': crop,
    'local': local,
    'select': select,
    'route': route,
    'reorg': reorg,
    'conv-extract': conv_extract,
    'extract': extract
}


def op_create(layer, inp, num, roof, feed, use_fp16=False):
    return op_types[layer.type](layer, inp, num, roof, feed, use_fp16=use_fp16)
