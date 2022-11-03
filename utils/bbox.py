# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : bbox.py
@Date    : 2022/10/29 23:35 
"""

import cv2
import math
import numpy as np
from enum import IntEnum, unique


@unique
class BboxFormat(IntEnum):
    corners = 1  # 使用4个角点表示一个四边形bbox
    xywh = 2  # 使用(x, y, w, h)格式表示一个长方形bbox. 其中(x, y)是长方形的左上角点, w和h分别代表长方形的宽和高
    diagonal = 3  # 使用(x1, y1, x2, y2)格式表示一个长方形bbox. 其中(x1, y1)代表长方形的左上角点, (x2, y2)代表右下角点


def minimal_rectangle(contours, ordered=True):
    contours = np.array(contours)
    min_rect = cv2.minAreaRect(contours)
    box_points = np.int0(cv2.boxPoints(min_rect))

    if ordered:
        box_points = order_clockwise_top_left_first(box_points)
    return box_points


def order_clockwise_top_left_first(box_points):
    assert len(box_points) == 4, "shape of points must be 4*2"

    # sort the points based on their x-coordinates
    x_sorted = box_points[np.argsort(box_points[:, 0]), :]

    # grab the left-most and right-most points from the sorted x-coordinate points
    left_most = x_sorted[:2, :]
    if left_most[0, 1] != left_most[1, 1]:
        left_most = left_most[np.argsort(left_most[:, 1]), :]
    else:
        left_most = left_most[np.argsort(left_most[:, 0])[::-1], :]
    tl, bl = left_most
    right_most = x_sorted[2:, :]
    if right_most[0, 1] != right_most[1, 1]:
        right_most = right_most[np.argsort(right_most[:, 1]), :]
    else:
        right_most = right_most[np.argsort(right_most[:, 0])[::-1], :]
    tr, br = right_most

    return np.array([tl, tr, br, bl], dtype=box_points.dtype)


def trans_integer(bbox, width=None, height=None):
    width = width or 1e12
    height = height or 1e12

    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
    x1, y1 = max(int(x1), 0), max(int(y1), 0)
    x2, y2 = min(math.ceil(x2), width), max(int(y2), 0)
    x3, y3 = min(math.ceil(x3), width), min(math.ceil(y3), height)
    x4, y4 = max(int(x4), 0), min(math.ceil(y4), height)

    return [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4],
    ]


def trans_xywh2corners(bbox, return_integer=True, width=None, height=None):
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y
    x3, y3 = x + w, y + h
    x4, y4 = x, y + h

    new_bbox = [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4],
    ]

    if return_integer and isinstance(x, float):
        new_bbox = trans_integer(new_bbox, width=width, height=height)

    return new_bbox


def trans_diagonal2corners(bbox, return_integer=True, width=None, height=None):
    x1, y1, x3, y3 = bbox
    x2, y2 = x3, y1
    x4, y4 = x1, y3

    new_bbox = [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4],
    ]

    if return_integer and isinstance(x1, float):
        new_bbox = trans_integer(new_bbox, width=width, height=height)

    return new_bbox
