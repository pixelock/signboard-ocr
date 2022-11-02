# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : manager.py
@Date    : 2022/10/17 22:00 
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm

from utils.image import get_rotate_crop_image
from utils.bbox import minimal_rectangle


class DataManager(object):
    def __init__(self, is_detection, is_recognition, is_scatter):
        self.is_detection = is_detection
        self.is_recognition = is_recognition
        self.is_scatter = is_scatter

        self.det_gts = None
        self.rec_gts = None

    def initialize(self):
        if self.is_detection:
            self.det_gts = self.get_detection_gts()
        if self.is_recognition:
            self.rec_gts = self.get_recognition_gts()

    def get_detection_gts(self):
        raise NotImplementedError

    def detection_output_paddle(self, output_dir):
        gts = self.det_gts or self.get_detection_gts()
        lines = []
        for path, gt in tqdm(gts.items()):
            line = '{}\t{}'.format(
                path,
                json.dumps(gt, ensure_ascii=False),
            )
            lines.append(line)

        with open(output_dir, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def save_cropped_image(self, output_dir, v2h=True, v2h_threshold=1.5):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gts = self.rec_gts

        for path, boxes in tqdm(gts.items()):
            img_data = cv2.imread(path, cv2.IMREAD_COLOR)

            for name, box in boxes.items():
                points = box['points']
                cropped_image = get_rotate_crop_image(
                    img_data,
                    np.array(points).astype('float32'),
                    v2h=v2h,
                    v2h_threshold=v2h_threshold,
                )
                cropped_path = os.path.join(output_dir, name)
                cv2.imwrite(cropped_path, cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def generate_cropped_image_name(self, prefix, index, image_name=None, box_info=None):
        raise NotImplementedError

    def get_recognition_gts(self):
        raise NotImplementedError

    def recognition_output_paddle(self, output_dir):
        gts = self.rec_gts or self.get_recognition_gts()
        lines = []
        for path, boxes in tqdm(gts.items()):
            for name, box in boxes.items():
                text = box['transcription']
                cropped_path = os.path.join(output_dir, name)
                line = '{}\t{}'.format(cropped_path, text)
                lines.append(line)

        with open(output_dir, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


class ReCTSManager(DataManager):
    NUM_SAMPLES = 20000
    P_IMAGE = 'train_ReCTS_{:0>6d}.jpg'
    P_GT = 'train_ReCTS_{:0>6d}.json'

    def __init__(self, path, level='lines'):
        super(ReCTSManager, self).__init__(
            is_detection=True,
            is_recognition=True,
            is_scatter=True,
        )

        self.root_path = path
        self.img_path = os.path.join(self.root_path, 'img')
        self.gt_path = os.path.join(self.root_path, 'gt')
        self.level = level

        self.initialize()

    def get_detection_gts(self):
        ground_truths = dict()
        for i in tqdm(range(1, self.NUM_SAMPLES + 1)):
            image_path = os.path.join(self.img_path, self.P_IMAGE.format(i))
            gt_path = os.path.join(self.gt_path, self.P_GT.format(i))

            with open(gt_path, 'r', encoding='utf-8') as f:
                raw_gt = json.load(f)
            boxes = raw_gt[self.level]

            available_boxes = []
            for index, box in enumerate(boxes):
                if box['ignore'] == 0:
                    text = box['transcription']
                    points = self.get_points(box['points'])
                    available_boxes.append({
                        'transcription': text,
                        'points': points
                    })
            ground_truths[image_path] = available_boxes
        return ground_truths

    def get_recognition_gts(self):
        ground_truths = dict()

        for image_path, gt in tqdm(self.det_gts.items()):
            image_name = os.path.splitext(os.path.split(image_path)[1])[0]
            available_boxes = dict()

            for index, box in enumerate(gt):
                text = box['transcription']
                points = box['points']
                cropped_name = self.generate_cropped_image_name('rects', index, image_name)
                available_boxes[cropped_name] = {
                    'transcription': text,
                    'points': points
                }

            ground_truths[image_path] = available_boxes

        return ground_truths

    def generate_cropped_image_name(self, prefix, index, image_name=None, box_info=None):
        real_name = os.path.splitext(image_name)[0]
        return '{}_{}_{}.jpg'.format(prefix, real_name, index)

    @staticmethod
    def get_points(p):
        return [[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]]


class LSVTManager(DataManager):
    def __init__(self, path):
        super(LSVTManager, self).__init__(
            is_detection=True,
            is_recognition=True,
            is_scatter=False,
        )
        self.root_path = path
        self.image_dir = os.path.join(path, 'train_images')
        self.gt_path = os.path.join(path, 'train_full_labels.json')

        with open(self.gt_path, 'r', encoding='utf-8') as f:
            self.raw_gt = json.load(f)

        self.initialize()

    def get_detection_gts(self):
        ground_truths = dict()
        for name, gt in tqdm(self.raw_gt.items()):
            image_path = os.path.abspath(os.path.join(self.image_dir, '{}.jpg'.format(name)))

            available_boxes = []
            for box in gt:
                if not box['illegibility']:
                    text = box['transcription']
                    points = box['points']
                    if len(points) > 4:
                        new_points = minimal_rectangle(points, ordered=True)
                        points = new_points.tolist()
                    available_boxes.append({
                        'transcription': text,
                        'points': points
                    })
            ground_truths[image_path] = available_boxes

        return ground_truths

    def get_recognition_gts(self):
        ground_truths = dict()

        for name, _ in tqdm(self.raw_gt.items()):
            image_path = os.path.abspath(os.path.join(self.image_dir, '{}.jpg'.format(name)))

            available_boxes = dict()
            gt = self.det_gts[image_path]
            for index, box in enumerate(gt):
                text = box['transcription']
                points = box['points']
                cropped_name = self.generate_cropped_image_name('lsvt', index, name)
                available_boxes[cropped_name] = {
                    'transcription': text,
                    'points': points,
                }

            ground_truths[image_path] = available_boxes

        return ground_truths

    def generate_cropped_image_name(self, prefix, index, image_name=None, box_info=None):
        return '{}_{}_{}.jpg'.format(prefix, image_name, index)


class ShopSignManager(DataManager):
    def __init__(self, path):
        super(ShopSignManager, self).__init__(
            is_detection=True,
            is_recognition=True,
            is_scatter=True,
        )

        self.root_path = path
        self.anno_path = os.path.join(self.root_path, 'annotation')
        self.initialize()

    def get_detection_gts(self):
        ground_truths = dict()

        name_list = [name for name in os.listdir(self.root_path) if name.lower().endswith('.jpg')]
        for name in tqdm(name_list):
            image_path = os.path.abspath(os.path.join(self.root_path, name))

            index = os.path.splitext(name)[0].replace('image_', '')
            if int(index) >= 21000:
                gt_name = os.path.join(self.anno_path, 'gt_img_{}.txt'.format(index))
                with open(gt_name, 'r', encoding='ANSI') as f:
                    lines = [t.strip()for t in f.read().split('\n') if t.strip()]
            else:
                gt_name = os.path.join(self.anno_path, 'image_{}.txt'.format(index))
                with open(gt_name, 'r', encoding='utf-8') as f:
                    lines = [t.strip()for t in f.read().split('\n') if t.strip()]

            available_boxes = []
            for line in lines:
                segs = line.split(',')
                assert len(segs) == 9, "error format sample: {}".format(line)

                text = segs[-1]
                if text == '###':
                    continue

                unrecognizable = False
                if '###' in text:
                    unrecognizable = True
                    text = text.replace('###', '').strip()

                points = self.get_points([int(p) for p in segs[:8]])
                available_boxes.append({
                    'transcription': text,
                    'points': points,
                    'unrecognizable': unrecognizable,
                })

            ground_truths[image_path] = available_boxes

        return ground_truths

    def get_recognition_gts(self):
        ground_truths = dict()

        for image_path, gt in tqdm(self.det_gts.items()):
            image_name = os.path.splitext(os.path.split(image_path)[1])[0]
            available_boxes = dict()

            for index, box in enumerate(gt):
                text = box['transcription']
                points = box['points']
                cropped_name = self.generate_cropped_image_name('shopsign', index, image_name)
                available_boxes[cropped_name] = {
                    'transcription': text,
                    'points': points
                }

            ground_truths[image_path] = available_boxes

        return ground_truths

    def generate_cropped_image_name(self, prefix, index, image_name=None, box_info=None):
        real_name = os.path.splitext(image_name)[0]
        return '{}_{}_{}.jpg'.format(prefix, real_name, index)

    @staticmethod
    def get_points(p):
        return [[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]]


if __name__ == '__main__':
    manager = ShopSignManager('../../data/ShopSign')
