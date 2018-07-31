"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda, Activation, Reshape, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import os

from keras.utils.training_utils import multi_gpu_model
from keras_applications.mobilenet_v2 import MobileNetV2

from keras_preprocessing.image import load_img
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tensorflow.core.framework.types_pb2 import DataType

from fcn.model1 import fcn_impossable, fcn_loss_impossible, fcn, weighted_classification_loss
import argparse as ap

from fcn.utils import rand
from train_fcn import sample_by_response


def _main():
    parser = ap.ArgumentParser()
    #argument_default="/Users/huanghaihun/PycharmProjects/keras-yolo3/data/xl_part1/*/*.jpg"
    parser.add_argument('--data_path',  help='data path string', type=str,
                        default="/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/produce_img/破边/*.jpg")
    parser.add_argument('--weight', help='weight_file ', type=str,
                        default="/tmp/logs/000/ep003-loss10.568-val_loss152353.547.h5")
    parser.add_argument('--batch_size', help='log dir ', type=int,
                        default=1)
    parser.add_argument('--tv_ratio', help='log dir ', type=float,
                        default=0.9)
    args = parser.parse_args()

    with tf.device("/cpu:0"):
        model = create_model(2)

    # yLabel = Input(shape=(2,))

    # model = multi_gpu_model(model,gpus=[0,1])


    val_split = args.tv_ratio
    import glob
    lines = glob.glob(args.data_path)
    lines = sample_by_response(lines, {"normal": 0.1})

    import numpy as np
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    lines = lines[:55]

    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    collector = []

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        # model.compile(optimizer=Adam(lr=1e-3, ), loss='categorical_crossentropy', metrics=['accuracy'])
        batch_size = args.batch_size
        print('evluate on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.load_weights(args.weight)
        import math
        pred_list, y_list, fid_list = model.predict_generator(data_generator_wrapper(lines[:num_train], batch_size, num_class=2, is_train=False),steps=math.ceil(1*num_train/batch_size))
        # pred_list, y_list, fid_list = model.predict_generator(data_generator_wrapper(lines[num_train:], batch_size, num_class=2, is_train=False),steps=math.ceil(1*num_val/batch_size))

        # metric = model.evaluate_generator(data_generator_wrapper(lines[num_train:], batch_size, num_class=2, is_train=False),steps=1)
        print(collector)

        # import numpy as np
        import pandas as pd

        pred_list = pred_list[:, 1]
        y_list = y_list[:, 1]
        fn = "/tmp/xx"
        ds = pd.DataFrame(data=[pred_list,y_list,fid_list[:,0]])
        ds.transpose().to_csv(fn)

        roc_auc = roc_auc_score(y_list, pred_list)
        avg_prec = average_precision_score(y_list, pred_list)
        print("roc_auc {}, avg_prec {} ".format(roc_auc, avg_prec))





def test():
    log_dir = '/tmp/logs/000/'

    val_split = 0.1
    import glob
    lines = glob.glob("/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/produce_img/*/*.jpg")

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    a = data_generator_wrapper(lines[:num_train], 2)
    for x in a:
        if isinstance(x[0], np.ndarray):
            print(x.shape)
        else:
            print(x)


def test_model(model, batch_size=2):
    a = model.predict(
        [np.ones([batch_size, 512, 512, 3]),
         np.ones([batch_size, 2]),
         np.ones([batch_size]),
         ], 2)
    print([x.shape for x in a])


def save_model_png(model, to_file='/tmp/model.png'):
    from keras.utils import plot_model
    plot_model(model, to_file=to_file)

# def create_model(num_classes=2,):
#     K.clear_session()  # get a new session
#     image_input = Input(shape=(512, 512, 3))
#     y_true = [Input(shape=(num_classes,)),
#               Input(shape=(1,)),]
#     model = MobileNetV2( include_top=False, weights=None, input_tensor=image_input)
#     x = Reshape([-1])(model.output)
#     x = Dense(num_classes)(x)
#
#     loss = Lambda(weighted_classification_loss, output_shape=(1,), name="loss")([x, *y_true])
#     model = Model([image_input, *y_true], loss)
#     return model
def create_model(num_classes=2,):
    K.clear_session()  # get a new session
    image_input = Input(shape=(512, 512, 3))

    label = Input(shape=(num_classes,))
    fid = Input(shape=(1,),dtype=tf.string)

    model = MobileNetV2(include_top=False, weights=None, input_tensor=image_input)
    x = Reshape([-1])(model.output)
    x = Dense(num_classes)(x)

    output = Activation('sigmoid')(x)
    model = Model([image_input,label,fid], [output, label, fid])
    return model

def create_impossible_model(num_classes=2, ):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))

    y_true = [Input(shape=(num_classes,)),
              Input(shape=(None, num_classes), ),
              Input(shape=(None, 1), ),
              ]
    model_body = fcn_impossable(image_input, num_classes)

    model_loss = Lambda(fcn_loss_impossible, output_shape=(1,1), name="fcn_loss_impossible", )([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    # test_model(model)
    # test_model(model_body)
    return model


import image_utils as iu


def data_generator_wrapper(annotation_lines, batch_size, num_class, is_train):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, num_class, is_train )


def get_one_hot(targets, nb_classes):
    if not isinstance(targets, np.ndarray):
        raise ValueError("targets should be ndarray")
    if targets.dtype not in (np.int, np.int32, np.int64):
        raise TypeError("targets dtype should be integer")
    res = np.eye(nb_classes, dtype=np.float32)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def data_generator(annotation_lines, batch_size, num_classes=2, is_train=True, drop_pos=0.1):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0

    def _one_hot(data):
        return get_one_hot(data, num_classes)

    while True:
        image_data = []
        label_data = []
        weight_data = []
        fid_data = []
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)

            fn = annotation_lines[i]
            image = load_img(fn)
            fid = fn.split("/")[-1][:23]
            image = iu.random_image(image, (512,512), random=is_train, )
            image_data.append(image)

            label = 0 if "flawInbox" in fn else 1

            if "flawInbox" in fn:
                weight = 1.0
            elif "normal" in fn:
                weight = 1.
            else: weight = 0.01
            fid_data.append(fid)
            label_data.append(label)
            weight_data.append(weight)
            i = (i + 1) % n
        image_data = np.array(image_data)

        image_data, label_data, weight_data, fid_data \
            = [np.stack(x) for x in [image_data, label_data, weight_data, fid_data]]

        arr = [image_data, _one_hot(label_data), fid_data]

        yield arr, np.zeros(batch_size)


if __name__ == '__main__':
    # a = create_model(2)
    # test_model(a,2)
    # _main()
    # test()
    _main()