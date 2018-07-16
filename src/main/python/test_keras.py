import keras
import numpy as np
from PIL import ImageDraw
import os
import tensorflow as tf
import tensorflow.python.layers
from keras import layers, Model, Input
from keras import Sequential
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras import regularizers

from keras import backend as K
from keras.optimizers import SGD

K.set_image_data_format('channels_first')

# Load the VGG gen_model
from src.main.python.image_utils import read_img, stack_img

# # vgg_model: Model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
# # # Load the Inception_V3 gen_model
# # inception_model = inception_v3.InceptionV3(weights='imagenet',include_top=False)
# #
# # # Load the ResNet50 gen_model
# # resnet_model = resnet50.ResNet50(weights='imagenet',include_top=False)
# #
# # Load the MobileNet gen_model
# # mobilenet_model = mobilenet.MobileNet(weights='imagenet',include_top=False)
#
# # vgg_model.summary()

input_shape = (3, 2560, 1920)
input_tensor_m = Input(shape=input_shape)
input_tensor_same = Input(shape=input_shape)
input_tensor_diff = Input(shape=input_shape)
input_tensor_define = Input(shape=input_shape)

model: Model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor_define)

x = layers.Flatten()(model.layers[-1].output)
gen_model = Model(input_tensor_define, x)

model_m = gen_model(input_tensor_m)
model_same = gen_model(input_tensor_same)
model_diff = gen_model(input_tensor_diff)
from keras import losses



def input_img(fn):
    # fn = 'zhadong/J01_2018.06.13 13_58_14'
    img = read_img(fn)
    print(img.shape)
    img = stack_img(img)
    print(img.shape)
    return img

def margin(big, small):
    return K.mean(K.maximum(0.5 + small - big, 0.), axis=-1)


def mse(x1, x2):
    x = layers.Subtract()([x1, x2])
    x = layers.Lambda(lambda _x: K.mean(K.square(_x), axis=[1,-1]))(x)
    return x


loss_same = mse(model_same, model_m)
loss_diff = mse(model_diff, model_m)
model_all = Model(inputs=[
    input_tensor_m,
    input_tensor_same,
    input_tensor_diff,
], outputs=[loss_same, loss_diff])

model_all.compile(
    SGD(nesterov=True, momentum=0.1),
    losses.hinge,
    target_tensors=[loss_diff, loss_same]
)
gen_model = Model(inputs=[input_tensor_m, input_tensor_same], outputs=loss_same)

data_m = input_img('diaowei/J01_2018.06.13 13_25_43')
data_same = input_img('diaowei/J01_2018.06.13 13_31_01')
data_diff = input_img('normal/J01_2018.06.13 13_24_39')

# model_all.predict([data_m, data_same, data_diff])
model_all.fit(x=[data_m, data_same, data_diff])

