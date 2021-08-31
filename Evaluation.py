import numpy as np
from AnchorUtils import norm_img
from AnnotationRect import iou, AnnotationRect
import tensorflow as tf
import os
from os import listdir
from DatasetMMP import MMP_Dataset_Evaluation


def call_eval_script():
    os.system('python eval_detections.py --detection detections.txt --dset_basedir ./dataset_mmp/val/')


def write_detections(filenames, boxes, scores):
    with open('detections.txt', 'a') as file:
        for filename in filenames:
            for (box, score) in zip(boxes, scores):
                fn = filename.numpy().decode('utf-8')[-12:]
                file.write(fn + ' 0 '
                           + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3])
                           + ' ' + str(score) + '\n')
        file.close()


def nms(boxes, scores, threshhold):
    rows, cols, factor, scale, rect = np.shape(boxes)
    boxes = np.reshape(boxes, (rows * cols * factor * scale, rect))
    scores = np.reshape(scores.numpy(), (rows * cols * factor * scale))
    D = np.empty((0, 4), dtype=np.int)
    S = np.empty(0)
    length = len(boxes)
    while length > 0:
        m = np.argmax(scores)
        if scores[m] == 0:
            break
        M = boxes[m]

        D = np.append(D, [M], axis=0)
        boxes = np.delete(boxes, m, axis=0)

        S = np.append(S, scores[m])
        scores = np.delete(scores, m, axis=0)
        length -= 1
        i = 0
        while i < length:
            ratio = iou(AnnotationRect.fromarray(M), AnnotationRect.fromarray(boxes[i]))
            if ratio >= threshhold:
                boxes = np.delete(boxes, i, axis=0)
                length -= 1
                scores = np.delete(scores, i, axis=0)
            else:
                i += 1
    return D, S


def evaluate_net(net, grid, dataset_path):
    if os.path.exists("detections.txt"):
        os.remove("detections.txt")
    batch_size = 16
    dataset = MMP_Dataset_Evaluation(dataset_path,
                          batch_size=batch_size,
                          num_parallel_calls=2,
                          anchor_grid=grid,
                          threshhold=0.9)
    print("Starting evaluation")
    for (filenames, imgs) in dataset():
        net_output = net(imgs, training=False)
        net_output = tf.reshape(net_output, (filenames.shape[0], 10, 10, 4, 3, 2))

        softmax_between_person_and_background = tf.nn.softmax(net_output)
        person_scores = softmax_between_person_and_background[:, :, :, :, :, 1]
        box_score_tuples = [nms(grid, max, 0.3) for max in person_scores]
        for (bestBoxes, bestScores) in box_score_tuples:
            write_detections(filenames, bestBoxes, bestScores)
    print("Evaluation completed")


def list_filenames_gen(path):
    print("Reading filenames")
    for file in listdir(path):
        if file.endswith('.jpg'):
            yield path + file
        else:
            continue
