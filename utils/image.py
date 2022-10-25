# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : image.py
@Date    : 2022/10/9 22:53 
"""

import cv2
import numpy as np


def get_rotate_crop_image(img, points, v2h=True, v2h_threshold=1.5):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)

    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if v2h and dst_img_height / dst_img_width >= v2h_threshold:
        dst_img = np.rot90(dst_img)

    return dst_img
