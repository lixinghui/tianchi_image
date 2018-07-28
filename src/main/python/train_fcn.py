"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
from fcn.model1 import yolo_body, fcn_loss
import argparse as ap


def _main():
    parser = ap.ArgumentParser()
    #argument_default="/Users/huanghaihun/PycharmProjects/keras-yolo3/data/xl_part1/*/*.jpg"
    parser.add_argument('--data_path',  help='data path string', type=str,
                        default="/Users/huanghaihun/PycharmProjects/keras-yolo3/data/xl_part1/*/*.jpg")
    parser.add_argument('--log_dir', help='log dir ', type=str,
                        default="/tmp/logs/000/")
    parser.add_argument('--batch_size', help='log dir ', type=int,
                        default=32)
    args = parser.parse_args()

    log_dir = args.log_dir
    model = create_model(2)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    import glob
    lines = glob.glob(args.data_path)

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom fcn_loss Lambda layer.
            'fcn_loss': lambda y_true, y_pred: y_pred})

        batch_size = args.batch_size
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, ),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')


def test():
    log_dir = '/tmp/logs/000/'

    val_split = 0.1
    import glob
    lines = glob.glob("/Users/huanghaihun/PycharmProjects/keras-yolo3/data/xl_part1/*/*.jpg")

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
        [np.ones([batch_size, 2560, 1920, 3]),
         np.ones([batch_size, 2]),
         np.ones([batch_size, 5 * 4, 2]),
         np.ones([batch_size, 5 * 4, 2])
         ], 2)
    print([x.shape for x in a])


# def test_model(model, batch_size=1):
#     a = model.predict(
#         [np.ones([batch_size, 2560, 1920, 3]),
#          ], 2)
#     print([x.shape for x in a])

def save_model_png(model, to_file='/tmp/model.png'):
    from keras.utils import plot_model
    plot_model(model, to_file=to_file)


def create_model(num_classes=2, ):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))

    y_true = [Input(shape=(num_classes,)),
              Input(shape=(None, num_classes), ),
              Input(shape=(None, 1), ),
              ]
    model_body = yolo_body(image_input, num_classes)

    model_loss = Lambda(fcn_loss, output_shape=(1,), name="fcn_loss", )([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    # test_model(model)
    # test_model(model_body)
    return model


import image_utils as iu


def data_generator_wrapper(annotation_lines, batch_size):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, )


def get_one_hot(targets, nb_classes):
    if not isinstance(targets, np.ndarray):
        raise ValueError("targets should be ndarray")
    if targets.dtype not in (np.int, np.int32, np.int64):
        raise TypeError("targets dtype should be integer")
    res = np.eye(nb_classes, dtype=np.float32)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def data_generator(annotation_lines, batch_size, cell_length=512, num_classes=2):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0

    def _one_hot(data):
        return get_one_hot(data, num_classes)

    while True:
        image_data = []
        label_data = []
        cell_label_data = []
        cell_weight_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            fn = annotation_lines[i][:-4]
            image = iu.read_img(fn, "WHC")
            image_data.append(image)
            label, cell_label, cell_weight = iu.gen_fcn_label(fn, length=cell_length)
            label_data.append(label)
            cell_label_data.append(cell_label)
            cell_weight_data.append(cell_weight)
            i = (i + 1) % n
        image_data = np.array(image_data)

        image_data, label_data, cell_label_data, cell_weight_data \
            = [np.stack(x) for x in [image_data, label_data, cell_label_data, cell_weight_data]]

        arr = [image_data, _one_hot(label_data), _one_hot(cell_label_data),
               np.reshape(cell_weight_data, [*cell_weight_data.shape, 1])]

        yield arr, np.zeros(batch_size)


if __name__ == '__main__':
    _main()
    # test()
    # model = create_model(num_classes=2)
