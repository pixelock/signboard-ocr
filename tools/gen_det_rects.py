# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : gen_det_rects.py
@Date    : 2022/10/9 21:50 
"""

import os
import sys
import cv2
import json
import numpy as np
from tqdm import tqdm

from constant import PATH_DATA, PATH_DATA_DET, PATH_DATA_RECOG, PATH_DATA_RECOG_IMG
from utils.image import get_rotate_crop_image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

DATASET_DIR = os.path.join(PATH_DATA, 'ICDAR-2019-ReCTS')
IMAGE_DIR = os.path.join(DATASET_DIR, 'img')
GT_DIR = os.path.join(DATASET_DIR, 'gt')


def run_det(det_level='lines'):
    """
    Generate ground truth for text detection task files with `PaddleOCR` framework format.

    :param det_level: `lines` or `chars`, default: `lines`. Detection level of text.
    """

    ground_truths = dict()
    for i in tqdm(range(1, 20001)):
        image_path = os.path.join(IMAGE_DIR, 'train_ReCTS_{:0>6d}.jpg'.format(i))
        gt_path = os.path.join(GT_DIR, 'train_ReCTS_{:0>6d}.json'.format(i))

        with open(gt_path, 'r', encoding='utf-8') as f:
            raw_gt = json.load(f)

        boxes = raw_gt[det_level]
        available_boxes = []
        for i, box in enumerate(boxes):
            if box['ignore'] == 0:
                text = box['transcription']
                p = box['points']
                points = [[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]]
                available_boxes.append({
                    'transcription': text,
                    'points': points
                })
        ground_truths[image_path] = available_boxes

    with open(os.path.join(PATH_DATA_DET, 'gt_rects.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(['{}\t{}'.format(k, json.dumps(v, ensure_ascii=False)) for k, v in ground_truths.items()]))


def run_recog(recog_level='lines', crop_image=True):
    """
    Generate ground truth file for text recognition task with `PaddleOCR` framework format.

    :param recog_level: `lines` or `chars`, default: `lines`. Detection level of text
    :param crop_image: whether save cropped text images
    """

    if crop_image:
        for i in tqdm(range(1, 20001)):
            image_path = os.path.join(IMAGE_DIR, 'train_ReCTS_{:0>6d}.jpg'.format(i))
            gt_path = os.path.join(GT_DIR, 'train_ReCTS_{:0>6d}.json'.format(i))

            with open(gt_path, 'r', encoding='utf-8') as f:
                raw_gt = json.load(f)
            image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)

            boxes = raw_gt[recog_level]

            for j, box in enumerate(boxes):
                if box['ignore'] == 0:
                    text = box['transcription']
                    p = box['points']
                    points = [[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]]
                    cropped_image = get_rotate_crop_image(image_data, np.array(points).astype('float32'))
                    cropped_path = os.path.join(PATH_DATA_RECOG_IMG, 'rects_{:0>6d}_{}.jpg'.format(i, j))
                    cv2.imwrite(cropped_path, cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    final_gts = dict()
    for i in tqdm(range(1, 20001)):
        gt_path = os.path.join(GT_DIR, 'train_ReCTS_{:0>6d}.json'.format(i))

        with open(gt_path, 'r', encoding='utf-8') as f:
            raw_gt = json.load(f)
        boxes = raw_gt[recog_level]

        for j, box in enumerate(boxes):
            if box['ignore'] == 0:
                text = box['transcription']
                cropped_path = os.path.join(PATH_DATA_RECOG_IMG, 'rects_{:0>6d}_{}.jpg'.format(i, j))
                final_gts[cropped_path] = text
    with open(os.path.join(PATH_DATA_RECOG, 'gt_rects.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(['{}\t{}'.format(k, v) for k, v in final_gts.items()]))


if __name__ == '__main__':
    run_det()
    run_recog(crop_image=True)
