# import keras
import numpy as np
from PIL import ImageDraw, Image
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.applications.imagenet_utils import decode_predictions
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path + "/../../..")

base = "data/xl_part1/"
output_prefix = "data/produce_img/"


def get_path(fn, post_fix): return os.path.join(base, fn) + post_fix


def read_bbox(fn):
    # matplotlib inline
    tree = ET.ElementTree(file=get_path(fn, ".xml"))
    a = []

    def collect(path):
        for x in tree.iterfind(path):
            a.append(int(x.text))

    collect("object/bndbox/")
    return np.asarray(a).reshape([-1, 4])


def draw_rectangle(draw, coordinates, color="black", width=10):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[2] + i, coordinates[3] + i)
        draw.rectangle((rect_start, rect_end), outline=color)


def draw_bb(fn):
    coord_tensor = read_bbox(fn)
    original = load_img(get_path(fn, ".jpg"))
    draw = ImageDraw.Draw(original)

    for i in range(coord_tensor.shape[0]):
        draw_rectangle(draw, coord_tensor[i, :])
    return original


def plot_show(fn):
    # load an image in PIL format
    original = draw_bb(fn)
    plt.imshow(original)
    plt.show()


def read_img(fn, mode='CWH'):
    x = load_img(get_path(fn, ".jpg"))
    # x = x.convert('LA')
    if mode == 'CWH':
        x = np.transpose(x, [2, 1, 0])
    else:
        x = np.asarray(x)
    return x


def stack_img(img):
    return np.stack([img])


def save_img(fn, data):
    img = Image.fromarray(np.uint8(data))
    img.save(fn + ".jpg")


def get_coord(horizontal_idx, verctical_idx,
              width=2560, height=1920,
              verctical_num=4, horizontal_num=5):
    xi, yi = width / horizontal_num * horizontal_idx, height / verctical_num * verctical_idx
    xa, ya = width / horizontal_num * (horizontal_idx + 1) - 1, height / verctical_num * (verctical_idx + 1) - 1
    return np.array([xi, yi, xa, ya])


def cut_image(img: np.ndarray, verctical_num=4, horizontal_num=5):
    hsplits = np.hsplit(img, horizontal_num)
    col = []
    for (horizontal_idx, x) in enumerate(hsplits):
        hsplit_of_vsplits = np.vsplit(x, verctical_num)
        for verctical_idx, y in enumerate(hsplit_of_vsplits):
            col.append((horizontal_idx, verctical_idx, y))

    return col


def preprocess(fn,
               need_draw_bb=True,
               output_prefix=output_prefix):
    img = None
    if need_draw_bb:
        img = draw_bb(fn)
    else:
        img = read_img(fn, 'WHC')
    img = np.asarray(img)
    all = cut_image(img)
    bbox_list = read_bbox(fn)

    for (i, j, data) in all:
        collector = []
        for bbox in bbox_list:
            part_img = get_coord(i, j)
            iou = get_iou(bbox, part_img)
            ios = get_ios(bbox, part_img)
            # is_negtive = all_belong(bbox, part_img)
            is_negtive = partial_belong(bbox, part_img) and ios > 0
            collector.append([is_negtive, ios])

        from functools import reduce
        is_negtive, ios = reduce(lambda x, y: (x[0] or y[0], max(x[1], y[1])), collector)
        if is_negtive:
            path = os.path.join(output_prefix, "{}_{}_{}_{}_{:.2f}".format(fn, i, j, is_negtive, ios))
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            save_img(path, data)


def all_belong(s, o):
    """
    if s belongs to o
    :param s:
    :param o:
    :return:
    """
    (s_xi, s_yi, s_xa, s_ya) = s
    (o_xi, o_yi, o_xa, o_ya) = o
    if s_xi >= o_xi and s_yi >= o_yi and s_xa <= o_xa and s_ya <= o_ya:
        return True
    return False


def partial_belong(s, o):
    """
    if s is partially belongs to o
    :param s:
    :param o:
    :return:
    """
    (s_xi, s_yi, s_xa, s_ya) = s
    (o_xi, o_yi, o_xa, o_ya) = o
    if (s_xi >= o_xi and s_xa <= o_xa) or (s_yi >= o_yi and s_ya <= o_ya):
        return True
    return False


def get_iou(s, o):
    (s_xi, s_yi, s_xa, s_ya) = s
    (o_xi, o_yi, o_xa, o_ya) = o

    x_a = max(s_xi, o_xi)
    y_a = max(s_yi, o_yi)

    x_b = min(s_xa, o_xa)
    y_b = min(s_ya, o_ya)
    interArea = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    area_a = (s_xa - s_xi + 1) * (s_ya - s_yi + 1)
    area_b = (o_xa - o_xi + 1) * (o_ya - o_yi + 1)
    iou = 1. * interArea / (area_a + area_b - interArea)
    return iou


def get_ios(s, o):
    """
    iou = 1. * interArea / area_s
    :param s:
    :param o:
    :return:
    """
    (s_xi, s_yi, s_xa, s_ya) = s
    (o_xi, o_yi, o_xa, o_ya) = o
    x_a = max(s_xi, o_xi)
    y_a = max(s_yi, o_yi)

    x_b = min(s_xa, o_xa)
    y_b = min(s_ya, o_ya)
    interArea = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    area_a = (s_xa - s_xi + 1) * (s_ya - s_yi + 1)
    # area_b = (o_xa - o_xi + 1) * (o_ya - o_yi + 1)
    iou = 1. * interArea / area_a
    return iou


# fn = 'diaowei/J01_2018.06.13 13_25_43'
# fn = 'diaowei/J01_2018.06.13 13_31_01'
# fn = 'diaowei/J01_2018.06.16 09_18_16'  # 只要切分框横/纵任一超过，就算是负样本，否则是正样本
# fn = 'maoban/J01_2018.06.16 08_47_24'  # 需要IOU > 50% 算作负样本
fn = 'zhadong/J01_2018.06.13 14_20_28'  # 需要IOU > 50% 算作负样本
# plot_show(fn)
if __name__ == '__main__':
    preprocess(fn)
