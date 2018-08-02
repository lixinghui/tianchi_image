'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras import Model
from keras.applications import resnet50
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K, Input, Model
import os
import numpy as np
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model

from save_util import MultiGPUCheckpointCallback

num_classes = 2
epochs = 200
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--data_path', help='data path string', type=str,
                    default="/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/bc")
parser.add_argument('--batch_size', help='batch_size ', type=int,
                    default=1)
parser.add_argument('--log_dir', help='log dir ', type=str,
                    default='output')
parser.add_argument('--model_tag', help='log dir ', type=str,
                    default="")
args = parser.parse_args()

model_name = 'keras_{}_trained_model.h5'.format(args.model_tag)
batch_size = args.batch_size
# x_train = np.load("/tmp/x.npy")
# y_train = np.load("/tmp/y.npy")
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


# target_size = (128, 128)
target_size = (256, 256)


# target_size = (197,197)

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true, curve="ROC")

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def create_model128():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(target_size[0], target_size[1], 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def create_model256():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(target_size[0], target_size[1], 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def create_resnet50():
    # not useful
    input_shape = (target_size[0], target_size[1], 3)
    vgg = resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape)

    model = Sequential()
    model.add(Flatten(input_shape=vgg.output_shape[1:]))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return Model(input=vgg.input, output=model(vgg.output))


def create_resnet():
    # not useful
    input_shape = (target_size[0], target_size[1], 3)
    vgg = VGG16(include_top=False, weights=None, input_shape=input_shape)

    model = Sequential()
    model.add(Flatten(input_shape=vgg.output_shape[1:]))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return Model(input=vgg.input, output=model(vgg.output))


def create_darknet(num_classes=2):
    input_shape = (target_size[0], target_size[1], 3)
    from fcn.model1 import darknet_6
    input_tensor = Input(input_shape)
    base_model = darknet_6(input_tensor)

    model = Sequential()
    model.add(Flatten(input_shape=(1, 1, 256)))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return Model(inputs=[input_tensor], outputs=[model(base_model)])


with tf.device("/cpu:0"):
    # model = create_model()
    # model = create_resnet50()
    # model = create_darknet(2)
    model_path = args.model_tag
    model = create_model256()
    # model = create_model128()
    if tf.gfile.Exists(model_path):
        model.load_weights(model_path)

model = multi_gpu_model(model, gpus=[0, 1])

# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.adam(lr=1e-4)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', auc_roc])

# model.save_weights("/tmp/test")
# model.load_weights("/tmp/test")
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    # model.fit(x_train[:10], y_train[:10],
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(x_train[:10], y_train[:10]),
    #           shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        # rescale=None,
        rescale=1. / 255,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.1)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)


    logging = TensorBoard(log_dir=args.log_dir)
    checkpoint = MultiGPUCheckpointCallback(args.log_dir + '/ep{epoch:03d}-loss{loss:.3f}-auc_roc{auc_roc:.3f}.h5',
                                            model,
                                            monitor='val_auc_roc', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc_roc', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_auc_roc', min_delta=0, patience=100, verbose=1)

    g_train = datagen.flow_from_directory(
        args.data_path,
        target_size=target_size,
        classes=['flaw', 'normal'],
        shuffle=True,
        batch_size=batch_size,
        subset='training'
    )
    g_valid = datagen.flow_from_directory(
        args.data_path,
        target_size=target_size,
        classes=['flaw', 'normal'],
        shuffle=True,
        batch_size=batch_size,
        subset='validation'
    )
    model.fit_generator(generator=g_train,
                        epochs=epochs,
                        validation_data=g_valid,
                        validation_steps=6,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping],
                        # validation_data=(x_train[:10], y_train[:10]),
                        workers=batch_size // 4)

    # model.fit_generator(datagen.flow(x_train[:10], y_train[:10],
    #                                  batch_size=4),
    #                     epochs=epochs,
    #                     validation_data=(x_train[:10], y_train[:10]),
    #                     workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
# model.save_weights(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
# scores = model.evaluate(x_train, y_train, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
