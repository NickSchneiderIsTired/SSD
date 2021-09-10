import tensorflow as tf
import os
from os import listdir
from DatasetMMP import MMP_Dataset_Evaluation
from constants import *


def call_eval_script():
    os.system('python eval_detections.py --detection detections.txt --dset_basedir dataset_mmp')


def write_detections(filename, boxes, scores):
    with open('detections.txt', 'a') as file:
        for (box, score) in zip(boxes, scores):
            fn = filename.numpy().decode('utf-8')[-12:]
            file.write(fn + ' 0 '
                       + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3])
                       + ' ' + str(score) + '\n')
        file.close()


def nms(boxes, scores, threshold):
    return tf.image.non_max_suppression(boxes, scores, iou_threshold=threshold, max_output_size=20)


def evaluate_net(net, grid, dataset_path, nms_threshold=0.3):
    if os.path.exists("detections.txt"):
        os.remove("detections.txt")
    batch_size = 16
    dataset = MMP_Dataset_Evaluation(dataset_path,
                          batch_size=batch_size,
                          num_parallel_calls=6,
                          anchor_grid=grid)
    flatted_grid = tf.reshape(grid, (GRID_X * GRID_Y * len(GRID_SIZES) * len(GRID_RATIOS), 4))
    flatted_scores_shape = GRID_X * GRID_Y * len(GRID_SIZES) * len(GRID_RATIOS)

    for (filenames, imgs) in dataset():
        net_output = net(imgs, training=False)
        net_output = tf.reshape(net_output, (filenames.shape[0], GRID_X, GRID_Y, len(GRID_SIZES), len(GRID_RATIOS), 2))

        softmax_between_person_and_background = tf.nn.softmax(net_output)
        person_scores = softmax_between_person_and_background[:, :, :, :, :, 1]

        for (filename, person_score) in zip(filenames, person_scores):
            flatted_scores = tf.reshape(person_score, flatted_scores_shape)
            selected_indices = nms(tf.cast(flatted_grid, dtype=tf.float32), flatted_scores, nms_threshold)

            selected_boxes = tf.gather(flatted_grid, selected_indices)
            selected_scores = tf.gather(flatted_scores, selected_indices)

            write_detections(filename, selected_boxes.numpy(), selected_scores.numpy())

    call_eval_script()


def list_filenames_gen(path):
    print("Reading filenames")
    for file in listdir(path):
        if file.endswith('.jpg'):
            yield path + file
        else:
            continue
