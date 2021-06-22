import numpy as np


class AnnotationRect:
    def __init__(self, x1, y1, x2, y2):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        self.x1 = x1 if x1 < x2 else x2
        self.y1 = y1 if y1 < y2 else y2
        self.x2 = x2 if x1 < x2 else x1
        self.y2 = y2 if y1 < y2 else y1

    def __str__(self):
        return 'x1: ' + str(self.x1) + ', y1: ' + str(self.y1) + ', x2: ' + str(self.x2) + ', y2: ' + str(self.y2)

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def area(self):
        return self.width() * self.height()

    def __array__(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @staticmethod
    def fromarray(arr):
        return AnnotationRect(arr[0], arr[1], arr[2], arr[3])


def area_intersection(rect1, rect2):
    rect1 = np.array(rect1)
    rect2 = np.array(rect2)
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    return x_overlap * y_overlap


def area_union(rect1, rect2):
    intersection = area_intersection(rect1, rect2)
    return rect1.area() + rect2.area() - intersection


def iou(rect1, rect2):
    intersection = area_intersection(rect1, rect2)
    union = area_union(rect1, rect2)
    return intersection / union


def get_highest_intersection(rect, gts=[]):
    max = 0
    for gt in gts:
        new_iou = iou(rect, AnnotationRect.fromarray(gt))
        if new_iou > max:
            max = new_iou
    return max

