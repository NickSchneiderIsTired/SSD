from AnnotationRect import AnnotationRect, get_highest_intersection
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


def create_negative_samples(label_grids, output_shape, negative_ratio):
    positive_values_count = tf.math.reduce_sum(label_grids)
    negative_samples = tf.random.uniform(output_shape, minval=0, maxval=1, dtype=tf.float32)

    negative_samples = tf.reshape(negative_samples, np.prod(list(output_shape)))
    negative_samples = tf.sort(negative_samples)
    negative_sample_count = positive_values_count * negative_ratio

    top_k = tf.math.top_k(negative_samples, k=negative_sample_count.numpy())
    negative_samples = negative_samples > top_k.values[-1].numpy()
    negative_samples = tf.cast(negative_samples, tf.float32)

    negative_samples = tf.reshape(negative_samples, output_shape)
    negative_samples = tf.add(negative_samples, tf.cast(label_grids, tf.float32))

    return tf.clip_by_value(negative_samples, clip_value_min=-1000000, clip_value_max=1)


def norm_img(img):
    img -= np.min(img)
    img /= np.max(img)
    return np.array(img * 255, dtype="uint8")


def unnormalize(array):
    min, max = np.min(array), np.max(array)
    res = (array - min) * 255 / (max - min)
    return res


def draw_rect(anchor_grid, label_grid, img, out):
    boxes = anchor_grid[label_grid.astype(bool)]
    img = norm_img(img)

    pil_img = Image.fromarray(img)

    img = draw_annotation_on_img(pil_img, boxes)
    img.save(out)


def draw_nms(bestBoxes, bestScores, image):
    for (box, score) in zip(bestBoxes, bestScores):
        if score > 0.01:
            image.rectangle(xy=box.tolist(), outline=128, width=2)
    return image


def draw_annotation_on_img(img, rectangles):
    draw = ImageDraw.Draw(img)
    for rectangle in rectangles:
        upper_left = (rectangle[0], rectangle[1])
        lower_right = (rectangle[2], rectangle[3])
        draw.rectangle((upper_left, lower_right), outline="yellow")
    return img


def create_label_grid(overlap_values, threshhold):
    return (overlap_values > threshhold).astype(int)  # int


def anchor_max_gt_overlaps(anchor_grid, gts=[]):
    rows, cols, scales, ratios, coords = np.shape(anchor_grid)
    res = np.empty((rows, cols, scales, ratios), dtype="float32")

    for row in range(rows):
        for col in range(cols):
            for scale in range(scales):
                for ratio in range(ratios):
                    res[row, col, scale, ratio] = get_highest_intersection(
                        AnnotationRect.fromarray(anchor_grid[row, col, scale, ratio]), gts)

    return res


def anchor_grid(fmap_rows,
                fmap_cols,
                scale_factor=1.0,
                scales=[],
                aspect_ratios=[]):
    res = np.empty((fmap_rows, fmap_cols, len(scales), len(aspect_ratios), 4), dtype=np.int)

    for row in range(fmap_rows):
        scaled_row = row * scale_factor + (scale_factor // 2)
        for col in range(fmap_cols):
            scaled_col = col * scale_factor + (scale_factor // 2)
            for scale_count, scale in enumerate(scales):
                half_scale = scale // 2
                for ratio_count, ratio in enumerate(aspect_ratios):
                    y1 = max(scaled_row - half_scale * ratio, 0)
                    x1 = max(scaled_col - half_scale, 0)
                    y2 = min(scaled_row + half_scale * ratio, 320)
                    x2 = min(scaled_col + half_scale, 320)
                    res[row, col, scale_count, ratio_count] = np.array([x1, y1, x2, y2])

    return res
