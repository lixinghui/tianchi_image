"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.applications import VGG16
from keras.layers import Input, Lambda, Reshape, Dense, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import os

from keras.utils.training_utils import multi_gpu_model

from keras_preprocessing.image import load_img
from sklearn.metrics import roc_auc_score, average_precision_score

from fcn.model1 import fcn_impossable, fcn_loss_impossible, fcn, weighted_classification_loss
import argparse as ap

from fcn.utils import rand

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def sample_by_response(file_list, table):
    def f(fn):
        for(k,v) in table.items():
            if k in fn:
                return rand() < v
        return True

    return [x for x in filter(f, file_list)]

def _main():
    parser = ap.ArgumentParser()
    #argument_default="/Users/huanghaihun/PycharmProjects/keras-yolo3/data/xl_part1/*/*.jpg"
    parser.add_argument('--data_path',  help='data path string', type=str,
                        default="/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/produce_img/破边/*.jpg")
    parser.add_argument('--log_dir', help='log dir ', type=str,
                        default="/tmp/logs/000/")
    parser.add_argument('--batch_size', help='log dir ', type=int,
                        default=1)
    parser.add_argument('--tv_ratio', help='log dir ', type=float,
                        default=0.0)
    parser.add_argument('--model', help='log dir ', type=str,
                        default="mobile")
    args = parser.parse_args()

    log_dir = args.log_dir
    if args.model == "mobile""":
        train_model, predict_model = create_model_mobile_train()
    elif args.model == "fcn":
        model = create_model(2)

    #with tf.device("/cpu:0"):
    #    model = create_model(2)
    #model = multi_gpu_model(model,gpus=[0,1])

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + args.model +'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = args.tv_ratio
    import glob
    import numpy as np

    lines = glob.glob(args.data_path)
    np.random.seed(10101)
    lines = sample_by_response(lines, {"normal": 0.07})

    lines = np.sort(lines)
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    lines = lines[:16] #TODO

    print('\n'.join(lines))
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    import keras.losses
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        train_model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy' )

        batch_size = args.batch_size
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        train_model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, num_class=2, is_train=False),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[:num_train], batch_size, num_class=2, is_train=False),
                            # validation_data=data_generator_wrapper(lines[num_train:], batch_size, num_class=2, is_train=False), #TODO
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        train_model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        # predict_model.load_weights(log_dir + 'trained_weights_stage_1.h5')

        xx = predict_model.predict_generator(data_generator_wrapper(lines[:num_train], batch_size, num_class=2, is_train=False),
                                        steps=np.math.ceil(1 * num_train / batch_size))
        pred_list, y_list = xx
        print(xx)

        def evaluate(pred_list, y_list):
            import pandas as pd
            pred_list = pred_list[:, 1]
            y_list = y_list[:, 1]
            fn = "/tmp/xx"
            ds = pd.DataFrame(data=[pred_list, y_list, ])
            ds.transpose().to_csv(fn)

            roc_auc = roc_auc_score(y_list, pred_list)
            avg_prec = average_precision_score(y_list, pred_list)
            print("roc_auc {}, avg_prec {} ".format(roc_auc, avg_prec))

        evaluate(pred_list, y_list)


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

def create_model(num_classes=2,):
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))

    y_true = [Input(shape=(num_classes,)),
              Input(shape=(1,)),]
    fcn_model = fcn(image_input, num_classes=num_classes)
    loss = Lambda(weighted_classification_loss,output_shape=(1,), name="loss")([fcn_model.output, *y_true])
    model = Model([image_input, *y_true], loss)
    return model

def create_model_mobile_train( ):
    K.clear_session()  # get a new session
    image_input = Input(shape=(512, 512, 3))
    y_true = Input(shape=(1,))
    model = VGG16( include_top=False, weights=None, input_tensor=image_input)
    x = Flatten()(model.output)
    x = Dense(128,activation='relu')(x)
    x = Dense(1)(x)

    # loss = Lambda(weighted_classification_loss, output_shape=(1,), name="loss")([x, *y_true])
    # train_model = Model([image_input, *y_true], loss)


    output = Activation('sigmoid')(x)
    train_model = Model([image_input], output)
    predict_model = Model([image_input, y_true], [output, y_true])


    return train_model, predict_model


def create_model_mobile_train_m(num_classes=2, ):
    K.clear_session()  # get a new session
    image_input = Input(shape=(512, 512, 3))
    y_true = Input(shape=(num_classes,))
    model = VGG16( include_top=False, weights=None, input_tensor=image_input)
    x = Flatten()(model.output)
    x = Dense(128,activation='relu')(x)
    x = Dense(num_classes)(x)

    # loss = Lambda(weighted_classification_loss, output_shape=(1,), name="loss")([x, *y_true])
    # train_model = Model([image_input, *y_true], loss)


    output = Activation('softmax')(x)
    train_model = Model([image_input], output)
    predict_model = Model([image_input, y_true], [output, y_true])


    return train_model, predict_model

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


def data_generator(annotation_lines, batch_size, num_classes=2, is_train=True):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0

    def _one_hot(data):
        return get_one_hot(data, num_classes)

    while True:
        image_data = []
        label_data = []
        weight_data = []
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            fn = annotation_lines[i]

            image = load_img(fn)
            image = iu.random_image(image, (512,512), random=is_train, )
            image_data.append(image)

            label = 0 if "flawInbox" in fn else 1
            if "flawInbox" in fn:
                weight = 1.0
            elif "normal" in fn:
                weight = 1.0
            else: weight = 0.01
            label_data.append(label)
            weight_data.append(weight)
            i = (i + 1) % n
        image_data = np.array(image_data)

        image_data, label_data, weight_data \
            = [np.stack(x) for x in [image_data, label_data, weight_data]]

        arr = [image_data, _one_hot(label_data), weight_data]

        yield image_data, label_data.astype(np.float32)


if __name__ == '__main__':
    # a = create_model(2)
    # test_model(a,2)
    # _main()
    # test()
    _main()
