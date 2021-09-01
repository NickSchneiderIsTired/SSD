import random
from augmentation import augment
import tensorflow as tf
import numpy as np
from AnnotationRect import AnnotationRect
from AnchorUtils import anchor_max_gt_overlaps, create_label_grid
from os import listdir


def read_groundtruth(file):
    with open(file) as f:
        lines = f.readlines()
        rects = []
        for line in lines:
            rects.append(AnnotationRect.fromarray(line.split()))
        return rects


def create_dict(path):
    dict = {}
    for file in listdir(path):
        if file.endswith('.jpg'):
            rects = read_groundtruth(path + file.split('.')[0] + '.gt_data.txt')
            dict[path + file] = rects

    return dict


def normalize(array):
    min, max = tf.math.reduce_min(array), tf.math.reduce_max(array)
    res = -1 + (2 / max - min) * (array - min)
    return res


class MMP_Dataset:
    def __init__(self, path_to_data, batch_size, num_parallel_calls, anchor_grid, threshhold):
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.files = create_dict(path_to_data)
        self.anchor_grid = anchor_grid
        self.threshhold = threshhold

    def data_gen(self):
        for filename, annotationRects in self.files.items():
            modifier = random.randint(1, 100)
            scores = anchor_max_gt_overlaps(self.anchor_grid, np.array([np.array(rect) for rect in annotationRects]))
            yield filename, create_label_grid(scores, self.threshhold), scores, modifier

    def __call__(self):
        dataset = tf.data.Dataset.from_generator(self.data_gen, output_types=(tf.string, tf.int32, tf.float32, tf.int32))
        # dataset = dataset.shuffle(buffer_size=len(self.files.keys()))
        dataset = dataset.repeat()
        dataset = dataset.map(self.load_single_example, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def load_single_example(self, filename, boxes, scores, modifier):
        img = tf.cast(tf.io.decode_png(tf.io.read_file(filename)), tf.float32)
        img = normalize(img)
        img = tf.image.pad_to_bounding_box(img, 0, 0, 320, 320)
        img, boxes, scores = augment(img, boxes, scores, modifier)
        return filename, img, boxes, scores

'''
    EVAL DATASET:
'''


def create_test_dict(path):
    return [path + file for file in listdir(path) if file.endswith('.jpg')]


class MMP_Dataset_Evaluation:
    def __init__(self, path_to_data, batch_size, num_parallel_calls, anchor_grid, threshhold):
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.files = create_test_dict(path_to_data)
        self.anchor_grid = anchor_grid
        self.threshhold = threshhold

    def data_gen(self):
        for filename in self.files:
            yield filename

    def __call__(self):
        dataset = tf.data.Dataset.from_generator(self.data_gen, output_types=tf.string)
        dataset = dataset.map(self.load_single_example, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def load_single_example(self, filename):
        img = tf.cast(tf.io.decode_png(tf.io.read_file(filename)), tf.float32)
        img = normalize(img)
        img = tf.image.pad_to_bounding_box(img, 0, 0, 320, 320)
        return filename, img
