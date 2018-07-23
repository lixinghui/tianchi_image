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

vgg_model: Model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
vgg_model = layers.Flatten()(vgg_model.layers[-1].output)
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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

batch_size = 1
dir_x = "/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/produce_img"
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
train_generator = datagen.flow_from_directory(
        dir_x,  # this is the target directory
        target_size=(448, 448),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')

vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=train_generator,
        validation_steps=800 // batch_size
)