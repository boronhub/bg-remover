import sys
import threading
import glob
from PIL import Image
import random
import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import keras
from keras.models import Model

from tiramisu.model import create_tiramisu
from camvid.mapping import map_labels


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', default='models/my_tiramisu.h5')

    parser.add_argument('--path_to_raw', default='camvid-master/CamVid/train/')

    parser.add_argument('--image_size', default=(360, 480))

    parser.add_argument('--path_to_labels',default='camvid-master/CamVid/trainannot/')

    parser.add_argument('--path_to_labels_list', default='camvid-master/CamVid/train.txt')

    parser.add_argument('--train_from_zero', type=bool, default=True)

    return parser.parse_args(args)


class BatchIndices(object):
    def __init__(self, n, bs, shuffle=False):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n)
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n: self.reset()
            ni = min(self.bs, self.n-self.curr)
            res = self.idxs[self.curr:self.curr+ni]
            self.curr += ni
            return res


class AugmentationGenerator(object):
    def __init__(self, x, y, bs=64, out_sz=(224, 224), train=True):
        self.x, self.y, self.bs, self.train = x, y, bs, train
        self.n, self.ri, self.ci, _ = x.shape
        self.idx_gen = BatchIndices(self.n, bs, train)
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape) == 4 else 1

    def get_slice(self, i, o):
        start = random.randint(0, i-o) if self.train else (i-o)
        return slice(start, start+o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.ri, self.ro)
        slice_c = self.get_slice(self.ci, self.co)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]

        if self.train and (random.random()>0.5):
            y = y[:, ::-1]
            x = x[:, ::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)
        items = (self.get_item(idx) for idx in idxs)
        xs, ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), -1, self.ych)


def load_image(fn, img_size):
    return np.array(Image.open(fn).resize(img_size, Image.NEAREST))


def load_data(path_to_raw, path_to_labels, img_size):

    image_files = glob.glob(path_to_raw + '*.png')
    images = np.stack([load_image(filename, img_size) for filename in image_files])

    label_files = glob.glob(path_to_labels + '*.png')
    labels = np.stack([load_image(filename, img_size) for filename in label_files])

    images = images / 255.
    images -= images.mean()
    images /= images.std()

    return images, labels


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    img_size = args.image_size

    raw, labels = load_data(args.path_to_raw, args.path_to_labels, img_size)

    if args.convert_from_camvid:
        labels = map_labels(args.path_to_labels_list, labels, img_size[1], img_size[0])

    n = len(raw)
    n_train = round(n*70/100)


    train_set = raw[:n_train]
    train_labels = labels[:n_train]
    val_set = raw[n_train:]
    val_labels = labels[n_train:]

    train_generator = AugmentationGenerator(train_set, train_labels, 1, train=True)
    test_generator = AugmentationGenerator(val_set, val_labels, 1, train=False)

    input_shape = (224, 224, 3)
    img_input = keras.layers.Input(shape=input_shape)
    x = create_tiramisu(32, img_input)
    model = Model(img_input, x)

    if not args.train_from_zero:
        model.load_weights(args.path_to_model_weights)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(1e-3, decay=1-0.99995), metrics=["accuracy"])


    model.fit_generator(train_generator, len(train_set), 20, verbose=2,
                        validation_data=test_generator, validation_steps=len(val_set))

    model.save_weights(args.output_path)


if __name__ == '__main__':
    main()




