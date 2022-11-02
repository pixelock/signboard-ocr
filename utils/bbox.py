# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : bbox.py
@Date    : 2022/10/29 23:35 
"""

import cv2
import numpy as np


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
