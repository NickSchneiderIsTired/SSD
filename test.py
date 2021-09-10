import tensorflow as tf
from AnchorUtils import anchor_grid
from Evaluation import evaluate_net
import os
from constants import *


def main(net=None, dataset_path="dataset_mmp/val/"):
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if os.path.exists("detections.txt"):
        os.remove("detections.txt")
    # Define necessary components
    grid = anchor_grid(GRID_X, GRID_Y, GRID_SCALE, GRID_SIZES, GRID_RATIOS)
    if net is None:
        net = tf.keras.models.load_model(r'models/test')
    evaluate_net(net, grid, dataset_path)


if __name__ == '__main__':
    main()
