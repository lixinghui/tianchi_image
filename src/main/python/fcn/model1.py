"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Cropping2D, Reshape, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from fcn.utils import compose

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks, padding=((1, 0), (1, 0)), strides=(2, 2), kernal=(3, 3)):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(padding)(x)
    x = DarknetConv2D_BN_Leaky(num_filters, kernal, strides=strides)(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x

def darknet_8(x):
    '''
    Darknent body having 52 Convolution2D layers
        处理padding, 奇数 strides=2 需要做 (1,1) padding
                    偶数 strides=2 需要做 (1,0) padding
                    2 -> 1 padding 0
                    1 -> 1 padding 0
        (1, 20, 15, 1024) ->(1, 10, 8, 1024), padding = ((1,0),(1,1))

    '''

    layer_stack = []
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1) #1
    x = resblock_body(x, 128, 2) #2
    x = resblock_body(x, 256, 4) #3
    x = resblock_body(x, 512, 4) #4
    x = resblock_body(x, 1024, 2) #5
    x = resblock_body(x, 1024, 2)   #;layer_stack.insert(0, x) #6
    x = resblock_body(x, 1024, 2, ) #;layer_stack.insert(0, x) #7
    x = resblock_body(x, 1024, 2, ) #;layer_stack.insert(0, x) #8
    x = resblock_body(x, 1024, 2, ) #;layer_stack.insert(0, x) #8
    return x

def darknet_6(x):
    '''
    Darknent body having 52 Convolution2D layers
        处理padding, 奇数 strides=2 需要做 (1,1) padding
                    偶数 strides=2 需要做 (1,0) padding
                    2 -> 1 padding 0
                    1 -> 1 padding 0
        (1, 20, 15, 1024) ->(1, 10, 8, 1024), padding = ((1,0),(1,1))

    '''

    layer_stack = []
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1) #1
    x = resblock_body(x, 128, 2) #2
    x = resblock_body(x, 256, 4) #3
    x = resblock_body(x, 512, 4) #4
    x = resblock_body(x, 1024, 2) #5
    x = resblock_body(x, 1024, 2)   #;layer_stack.insert(0, x) #6
    x = resblock_body(x, 1024, 2, ) #;layer_stack.insert(0, x) #7

    x = make_last_layers(x, 512, 256)
    return x

def fcn(inputs, num_classes=2):
    x = Model(inputs, darknet_8(inputs))
    x = Dense(num_classes)(x.output)
    x = Reshape([num_classes])(x)
    return Model(inputs, x)

def darknet_tolong(x):
    '''
    Darknent body having 52 Convolution2D layers
        处理padding, 奇数 strides=2 需要做 (1,1) padding
                    偶数 strides=2 需要做 (1,0) padding
                    2 -> 1 padding 0
                    1 -> 1 padding 0
        (1, 20, 15, 1024) ->(1, 10, 8, 1024), padding = ((1,0),(1,1))

    '''

    layer_stack = []
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 4)
    x = resblock_body(x, 512, 4)
    x = resblock_body(x, 1024, 2)
    x = resblock_body(x, 1024, 2)
    x = resblock_body(x, 1024, 2, ) #20 15
    x = resblock_body(x, 1024, 2, padding=((1,0),(1,1)))
    x = resblock_body(x, 1024, 2, padding=((1,0),(1,0)));layer_stack.insert(0, x) #5,4
    x = resblock_body(x, 1024, 2, padding=((1,1),(1,0)));layer_stack.insert(0, x) #3,2 = 5+1,4 / 2,2
    x = resblock_body(x, 1024, 2,
                      padding=((1, 1), (1, 0)), );layer_stack.insert(0, x)  # 2,1 = 3+1,2/2,2
    x = resblock_body(x, 1024, 2,
                      padding=((0, 0), (0, 0)), strides=(2, 1));layer_stack.insert(0, x)  # 1,1

    # [(1, 1, 1, 1024), (1, 2, 1, 1024), (1, 3, 2, 1024), (1, 5, 4, 1024), (1, 10, 8, 1024)]
    return layer_stack


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1))
    )(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)),
        Reshape([-1, out_filters])
    )(x)
    return y


def fcn_impossable(inputs, num_classes=2):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_tolong(inputs))
    layer_stack = darknet.output
    # print(layer_stack)
    y1 = DarknetConv2D(num_classes, (1, 1))(layer_stack[0])
    y1 = Reshape([num_classes])(y1)

    # [(1, 1, 1, 1024), (1, 2, 1, 1024), (1, 3, 2, 1024), (1, 5, 4, 1024), ]
    x = layer_stack[0]
    x = compose(
        DarknetConv2D_BN_Leaky(1024, (1, 1)),
        UpSampling2D(size=(2, 1)))(x)
    x = Concatenate()([x, layer_stack[1]])
    x = compose(
        DarknetConv2D_BN_Leaky(1024, (1, 1)),
        UpSampling2D(size=(2, 2)),
        Cropping2D(((0, 1), (0, 0))), )(x)
    x = Concatenate()([x, layer_stack[2]])

    x = compose(
        DarknetConv2D_BN_Leaky(1024, (1, 1)),
        UpSampling2D(size=(2, 2)),
        Cropping2D(((0, 1), (0, 0)))
    )(x)
    x = Concatenate()([x, layer_stack[3]])

    x = make_last_layers(x, 1024, num_classes)

    return Model(inputs, [y1,x,])
    # return Model(inputs, layer_stack)

def weighted_classification_loss(args, ):
    pred_1,  \
    y_1, \
    weight = args

    c1_loss = K.categorical_crossentropy(y_1, pred_1, True) * weight

    loss = K.mean(c1_loss,keepdims=True)
    return loss

def fcn_loss_impossible(args, ):
    pred_1, pred_5_4, \
    y_1, y_5_4, \
    weight = args

    # print(args)
    c1_loss = K.binary_crossentropy(y_1, pred_1, True)

    c20_15_loss = K.mean(K.binary_crossentropy(y_5_4, pred_5_4, True) * weight, axis=[1])

    loss = K.mean(c1_loss + c20_15_loss)
    return loss
