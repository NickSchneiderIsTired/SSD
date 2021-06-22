import numpy as np
import tensorflow as tf
from AnchorUtils import anchor_grid
from Evaluation import evaluate_net
import os
from DatasetMMP import MMP_Dataset_Test


def normalize(array):
    min, max = tf.math.reduce_min(array), tf.math.reduce_max(array)
    res = -1 + (2 / max - min) * (array - min)
    return res


def load_test_images(path):
    filenames = np.empty(0)
    images = np.empty(0)
    for file in os.listdir(path):
        img = tf.cast(tf.io.decode_png(tf.io.read_file(path + file)), tf.float32)
        img = normalize(img)
        img = tf.image.pad_to_bounding_box(img, 0, 0, 320, 320)

        np.append(filenames, [file], axis=0)
        np.append(images, [img], axis=0)

    return filenames, images


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if os.path.exists("detections.txt"):
        os.remove("detections.txt")
    # Define necessary components
    batch_size = 16
    grid = anchor_grid(10, 10, 32.0, [70, 100, 140, 200], [0.5, 1.0, 2.0])
    net = tf.keras.models.load_model(r'models/mobilenet')
    dataset = MMP_Dataset_Test("dataset_mmp/test/",
                               batch_size=batch_size,
                               num_parallel_calls=6,
                               anchor_grid=grid,
                               threshhold=0.5)
    counter = 0
    for (filenames, imgs) in dataset():
        res = net(imgs, training=False)
        res = tf.reshape(res, (tf.size(filenames), 10, 10, 4, 3, 2))
        evaluate_net(res, grid, imgs, filenames, 0)
        counter+=1
        print(filenames)


if __name__ == '__main__':
    main()
