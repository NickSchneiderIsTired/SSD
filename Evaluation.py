import numpy as np
from AnnotationRect import iou, AnnotationRect
import tensorflow as tf
from PIL import Image, ImageDraw
from AnchorUtils import unnormalize, draw_nms


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


def evaluate_net(net_output, grid, imgs, count):
    img = imgs[0]
    numpy_img = unnormalize(img.numpy()).astype('uint8')
    net_output = net_output[:, :, :, :, :, 1]
    max = tf.nn.softmax(net_output[0])
    bestBoxes, bestScores = nms(grid, max, 0.3)
    eval_img = Image.fromarray(numpy_img)
    draw_eval = ImageDraw.Draw(eval_img)
    draw_nms(bestBoxes, bestScores, draw_eval)
    eval_img.save(r'eval_img' + str(count) + '.jpg')
