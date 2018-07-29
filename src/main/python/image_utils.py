# import keras
import numpy as np
from PIL import ImageDraw, Image
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.applications.imagenet_utils import decode_predictions
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from fcn.utils import rand

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path + "/../../..")

base = "data/xl_part1/"
output_prefix = "data/produce_img/"

import glob


def get_path(fn, post_fix):
    return fn + post_fix


# def get_path(fn, post_fix): return os.path.join(base, fn) + post_fix


def read_bbox(fn):
    # matplotlib inline
    if os.path.exists(get_path(fn, ".xml")):
        tree = ET.ElementTree(file=get_path(fn, ".xml"))
        a = []

        def collect(path):
            for x in tree.iterfind(path):
                a.append(int(x.text))

        collect("object/bndbox/")
        return np.asarray(a).reshape([-1, 4])
    else:
        return None


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


def read_img(fn, mode='CHW'):
    x = load_img(get_path(fn, ".jpg"))
    if mode == 'CHW':
        x = np.transpose(x, [2, 1, 0])
    elif mode == "WHC":
        x = np.transpose(x, [1, 0, 2])
    else:
        x = np.asarray(x)

    return x


def stack_img(img):
    return np.stack([img])


def save_img(fn, data):
    img = Image.fromarray(np.transpose(np.uint8(data), [1,0,2]))
    img.save(fn + ".jpg")


# def get_coord(horizontal_idx, verctical_idx,
#               width=2560, height=1920,
#               verctical_num=8, horizontal_num=6):
#     xi, yi = width / horizontal_num * horizontal_idx, height / verctical_num * verctical_idx
#     xa, ya = width / horizontal_num * (horizontal_idx + 1) - 1, height / verctical_num * (verctical_idx + 1) - 1
#     return np.array([xi, yi, xa, ya])

def get_coord(horizontal_idx, verctical_idx, w_length, h_length,
              width=2560, height=1920, ):
    xi, yi = max(w_length * horizontal_idx, 0), max(h_length * verctical_idx, 0)
    xa, ya = min(w_length * (horizontal_idx + 1) - 1, width), min(height, h_length * (verctical_idx + 1) - 1)
    return np.array([xi, yi, xa, ya])


def cut_image(img: np.ndarray, w_length, h_length):
    w, h, _ = img.shape
    hsplits = np.vsplit(img, range(w_length, w, w_length))
    col = []
    for (horizontal_idx, x) in enumerate(hsplits):
        hsplit_of_vsplits = np.hsplit(x, range(h_length, h, h_length))
        for verctical_idx, y in enumerate(hsplit_of_vsplits):
            col.append((horizontal_idx, verctical_idx, y))

    return col


def random_image(image, input_shape, random=False, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    iw, ih = image.size
    h, w = input_shape

    if not random:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255.

        return image_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    return image_data


def preprocess_dir(glob_path, belong_type="ios", save_type="neg"):
    fn_list = glob.glob(glob_path)
    fn_list = [x[:-4] for x in fn_list]
    for fn in fn_list:
        preprocess(fn, need_draw_bb=True, belong_type=belong_type, save_type=save_type)


def preprocess(fn,
               need_draw_bb=True,
               output_prefix=output_prefix,
               ios_threshoud=0.1,
               belong_type="ios",
               save_type="neg"):
    img = None
    bbox_list = read_bbox(fn)

    if bbox_list is not None and need_draw_bb:
        img = draw_bb(fn)
    else:
        img = read_img(fn, 'WHC')
    img = np.asarray(img)
    w_length, h_length = 512, 480
    all = cut_image(img, w_length, h_length)

    saves = None
    if save_type == "neg":
        saves = [True]
    elif save_type == "pos":
        saves = [False]
    elif save_type == "both":
        saves = [True, False]
    else:
        raise ValueError("save_type must be in [neg, pos, both]")
    fun_negtive = None
    if belong_type == 'all':
        fun_negtive = all_belong
    elif belong_type == 'partial':
        fun_negtive = partial_belong
    elif belong_type == 'ios':
        fun_negtive = lambda x, y: True
    else:
        raise ValueError("belong_type not supported")

    def save(fn, data, flag, ios, i, j):
        fn = fn.split("/")[-2:]
        fn = os.path.join(*fn)
        path = os.path.join(output_prefix, "{}_{}_{}_{}_{:.2f}".format(fn, i, j, flag, ios))
        print("output_path: {}".format(path))
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        save_img(path, data)

    def produce_flaw(i, j, data, fn):
        collector = []
        for bbox in bbox_list:
            part_img = get_coord(i, j, w_length, h_length)
            # iou = get_iou(bbox, part_img)
            ios = get_ios(bbox, part_img)
            is_negtive = fun_negtive(bbox, part_img) and ios > ios_threshoud
            collector.append([is_negtive, ios])

        from functools import reduce
        is_negtive, ios = reduce(lambda x, y: (x[0] or y[0], max(x[1], y[1])), collector)

        flag = "flawInbox" if is_negtive else "flawOutbox"
        if is_negtive in saves:
            save(fn, data, flag, ios, i, j)

    def produce_norm(i, j, data, fn):
        save(fn, data, "normal", 0, i, j)

    for (i, j, data) in all:
        if bbox_list is not None:
            produce_flaw(i, j, data, fn)
        else:
            produce_norm(i, j, data, fn)

def gen_fcn_label(fn,
                  ios_threshoud=0.1,
                  belong_type="ios",
                  length=512,
                  ):
    """生成FCN的标注信息"""
    bbox_list = read_bbox(fn)

    fun_negtive = None
    if belong_type == 'all':
        fun_negtive = all_belong
    elif belong_type == 'partial':
        fun_negtive = partial_belong
    elif belong_type == 'ios':
        fun_negtive = lambda x, y: True
    else:
        raise ValueError("belong_type not supported")

    v_num = 5
    h_num = 4

    def produce_neg():
        img_collector = []
        for i in range(v_num):
            for j in range(h_num):
                bb_collector = []
                for bbox in bbox_list:
                    part_img = get_coord(i, j, length)
                    # iou = get_iou(bbox, part_img)
                    ios = get_ios(bbox, part_img)
                    is_negtive = fun_negtive(bbox, part_img) and ios > ios_threshoud
                    bb_collector.append([is_negtive, ios])

                from functools import reduce
                is_negtive, ios = reduce(lambda x, y: (x[0] or y[0], max(x[1], y[1])), bb_collector)

                img_collector.append(not is_negtive)
        # weight = map(lambda x: 1.0 if not x else 0.1, img_collector)
        weight = [1.0 if not x else 0.1 for x in img_collector]
        return 1, np.asarray(img_collector, dtype=np.int32), np.asarray(weight, dtype=np.float32)

    def produce_pos():
        return 0, np.ones(v_num * h_num, dtype=np.int32), np.ones((v_num * h_num), dtype=np.float32)

    return produce_pos() if bbox_list is None else produce_neg()


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


# plot_show(fn)

def preprocess_all_dir(path):
    dir_list = glob.glob(path)
    for x in dir_list:
        preprocess_dir(x, "ios", save_type="neg")


def validate_random_img(fn):
    a = load_img(get_path(fn, ".jpg"))
    plt.imshow(a)

    ra = random_image(a, (512, 512), random=True)
    ra = (ra * 255).astype(np.uint8)

    plt.imshow(ra)

    plt.show()


if __name__ == '__main__':
    # preprocess(fn, True,belong_type="partial",save_type="both")
    # dirx = "/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/part2/ȱγ"
    # dirx = "/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/part2/֯ϡ"
    # dirx = "/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/part2/修印"
    # dirx = "/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/part2/剪洞"
    # preprocess_all_dir("/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/part2/*")

    # fn = 'diaowei/J01_2018.06.13 13_25_43'
    # fn = 'diaowei/J01_2018.06.13 13_31_01'
    # fn = 'diaowei/J01_2018.06.16 09_18_16'  # 只要切分框横/纵任一超过，就算是负样本，否则是正样本
    # fn = 'maoban/J01_2018.06.16 08_47_24'  # 需要IOS > 50% 算作负样本
    # fn = '/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/xl_part1/diaowei/J01_2018.06.13 13_25_43'  # 需要IOS > 50% 算作负样本
    # fn = '/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/xl_part1/normal/J01_2018.06.13 13_23_08'

    # print(gen_fcn_label(fn))
    # read_img(fn)
    # validate_random_img('/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/produce_img/剪洞/J01_2018.06.22 08_45_25_2_1_True_1.00')
    preprocess_dir('/Users/huanghaihun/PycharmProjects/come_on_leg_man/data/part2/正常/*.jpg'
                   )
