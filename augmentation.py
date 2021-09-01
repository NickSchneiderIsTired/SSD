import random

import tensorflow as tf

"""
Assignment 08 - Data Augmentation

This file contains important functions for data augmentation.

Each mini-batch in the pipeline will access these functions to simulate a larger dataset
"""


def augment(img, boxes, scores, modifier):
    if modifier % 5 == 0:
        img, boxes, scores = flip(img, boxes, scores)
    if modifier % 7 == 0:
        img, boxes, scores = rotate(img, boxes, scores)
    if modifier % 18 == 0:
        img = noise(img)
    if modifier % 13 == 0:
        img, boxes, scores = crop(img, boxes, scores)

    return img, boxes, scores


def flip(img, boxes, scores):
    horizontally = random.random() > 0.9
    reshaped_boxes = tf.reshape(boxes, (10, 10, 12))
    reshaped_scores = tf.reshape(scores, (10, 10, 12))

    img = tf.image.flip_left_right(img) if horizontally else tf.image.flip_up_down(img)
    reshaped_boxes = tf.image.flip_left_right(reshaped_boxes) if horizontally else tf.image.flip_up_down(reshaped_boxes)
    reshaped_scores = tf.image.flip_left_right(reshaped_scores) if horizontally else tf.image.flip_up_down(reshaped_scores)
    return img, tf.cast(tf.reshape(reshaped_boxes, (10, 10, 4, 3)), tf.int32), tf.reshape(reshaped_scores, (10, 10, 4, 3))


def rotate(img, boxes, scores):
    ninety_deg_steps = random.randint(1, 3)  # 90, 180, 270°
    reshaped_boxes = tf.reshape(boxes, (10, 10, 12))
    reshaped_scores = tf.reshape(scores, (10, 10, 12))

    img = tf.image.rot90(img, ninety_deg_steps)
    reshaped_boxes = tf.image.rot90(reshaped_boxes, ninety_deg_steps)
    reshaped_scores = tf.image.rot90(reshaped_scores, ninety_deg_steps)
    return img, tf.cast(tf.reshape(reshaped_boxes, (10, 10, 4, 3)), tf.int32), tf.reshape(reshaped_scores, (10, 10, 4, 3))


# Get a random of x0.8 the region of the image and upscale to 320x320
# Boxes/scores.shape = 10x10x4x3
# img.shape = 320x320x3
def crop(img, boxes, scores):
    img = tf.image.central_crop(img, central_fraction=0.8)
    img = tf.image.resize(img, [320, 320])

    reshaped_boxes = tf.reshape(boxes, (10, 10, 12))
    reshaped_boxes = tf.image.central_crop(reshaped_boxes, central_fraction=0.8)
    reshaped_boxes = tf.image.resize(reshaped_boxes, [10, 10])

    reshaped_scores = tf.reshape(scores, (10, 10, 12))
    reshaped_scores = tf.image.central_crop(reshaped_scores, central_fraction=0.8)
    reshaped_scores = tf.image.resize(reshaped_scores, [10, 10])

    return img, tf.cast(tf.reshape(reshaped_boxes, (10, 10, 4, 3)), tf.int32), tf.reshape(reshaped_scores, (10, 10, 4, 3))


def noise(img):
    return tf.add(img, tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.1, dtype=tf.float32))
