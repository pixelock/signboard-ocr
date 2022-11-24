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
import collections
from tqdm import tqdm

from utils.image import get_rotate_crop_image
from utils.bbox import minimal_rectangle, trans_xywh2corners, trans_integer


class DataManager(object):
    def __init__(self, name, is_detection, is_recognition, is_scatter):
        self.dataset_name = name
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

    def get_recognition_gts(self):
        ground_truths = dict()

        for image_path, gt in tqdm(self.det_gts.items()):
            image_name = os.path.splitext(os.path.split(image_path)[1])[0]
            available_boxes = dict()

            for index, box in enumerate(gt):
                text = box['transcription']
                points = box['points']
                if '###' in text:
                    continue
                if not text.replace('#', '').strip():
                    continue
                if 'invalid' in box and box['invalid']:
                    continue
                cropped_name = self.generate_cropped_image_name(self.dataset_name, index, image_name)
                available_boxes[cropped_name] = {
                    'transcription': text,
                    'points': points
                }

            ground_truths[image_path] = available_boxes

        return ground_truths

    def generate_cropped_image_name(self, prefix, index, image_name=None, box_info=None):
        return '{}_{}_{}.jpg'.format(prefix, image_name, index)

    def recognition_output_paddle(self, output_dir, image_dir):
        gts = self.rec_gts or self.get_recognition_gts()
        lines = []
        for path, boxes in tqdm(gts.items()):
            for name, box in boxes.items():
                text = box['transcription']
                cropped_path = os.path.join(image_dir, name)
                line = '{}\t{}'.format(cropped_path, text)
                lines.append(line)

        with open(output_dir, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    @staticmethod
    def convert_points_format(p):
        return [[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]]


class ReCTSManager(DataManager):
    NUM_SAMPLES = 20000
    P_IMAGE = 'train_ReCTS_{:0>6d}.jpg'
    P_GT = 'train_ReCTS_{:0>6d}.json'

    def __init__(self, path, level='lines'):
        super(ReCTSManager, self).__init__(
            name='rects',
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
            image_path = os.path.abspath(os.path.join(self.img_path, self.P_IMAGE.format(i)))
            gt_path = os.path.join(self.gt_path, self.P_GT.format(i))

            with open(gt_path, 'r', encoding='utf-8') as f:
                raw_gt = json.load(f)
            boxes = raw_gt[self.level]

            available_boxes = []
            for index, box in enumerate(boxes):
                text = box['transcription'] if box['ignore'] == 0 else '###'
                points = self.convert_points_format(box['points'])
                available_boxes.append({
                    'transcription': text,
                    'points': points
                })
            ground_truths[image_path] = available_boxes
        return ground_truths


class LSVTManager(DataManager):
    def __init__(self, path):
        super(LSVTManager, self).__init__(
            name='lsvt',
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
                text = box['transcription'].replace('＃', '#') if not box['illegibility'] else '###'
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


class ShopSignManager(DataManager):
    def __init__(self, path, exclude_unrecognizable=False):
        self.exclude_unrecog = exclude_unrecognizable
        name = 'shopsign' if not self.exclude_unrecog else 'shopsign_ex_unrecog'

        super(ShopSignManager, self).__init__(
            name=name,
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
                with open(gt_name, 'r', encoding='gbk') as f:
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
                invalid = False
                if self.exclude_unrecog:
                    if text == '###':
                        continue
                    if '###' in text:
                        invalid = True
                        text = text.replace('###', '').strip()

                points = self.convert_points_format([int(p) for p in segs[:8]])
                available_boxes.append({
                    'transcription': text,
                    'points': points,
                    'invalid': invalid,
                })

            ground_truths[image_path] = available_boxes

        return ground_truths


class BDCIPOIManager(DataManager):
    def __init__(self, path):
        super(BDCIPOIManager, self).__init__(
            name='bdci_poi',
            is_detection=True,
            is_recognition=True,
            is_scatter=False,
        )

        self.root_path = path
        self.image_path_train = os.path.join(self.root_path, 'train_images')
        self.image_path_test_a = os.path.join(self.root_path, 'testA_images')
        self.image_path_test_b = os.path.join(self.root_path, 'testB_images')

        with open(os.path.join(self.root_path, 'public_train_label.json'), 'r', encoding='utf-8') as f:
            self.train_gts = json.load(f)['data']
        with open(os.path.join(self.root_path, 'public_test_A.json'), 'r', encoding='utf-8') as f:
            self.test_a_gts = json.load(f)['data']
        with open(os.path.join(self.root_path, 'public_test_B.json'), 'r', encoding='utf-8') as f:
            self.test_b_gts = json.load(f)['data']

        self.initialize()

    def get_detection_gts(self):
        ground_truths = dict()

        all_gts = dict()
        for _, value in self.train_gts.items():
            image_id = value['image_id']
            image_name = '{}.jpg'.format(image_id)
            image_path = os.path.join(self.image_path_train, image_name)
            all_gts[image_path] = value['texts']
        for _, value in self.test_a_gts.items():
            image_id = value['image_id']
            image_name = '{}.jpg'.format(image_id)
            image_path = os.path.join(self.image_path_test_a, image_name)
            all_gts[image_path] = value['texts']
        for _, value in self.test_b_gts.items():
            image_id = value['image_id']
            image_name = '{}.jpg'.format(image_id)
            image_path = os.path.join(self.image_path_test_b, image_name)
            all_gts[image_path] = value['texts']

        for path, gts in tqdm(all_gts.items()):
            image_path = os.path.abspath(path)
            available_boxes = []
            for box in gts:
                text = box['text']
                contour = box['contour']
                points = minimal_rectangle(contour, ordered=True).tolist()
                available_boxes.append({
                    'transcription': text,
                    'points': points,
                })

            ground_truths[image_path] = available_boxes

        return ground_truths


class CTWManager(DataManager):
    def __init__(self, path):
        super(CTWManager, self).__init__(
            name='ctw',
            is_detection=True,
            is_recognition=True,
            is_scatter=False,
        )

        self.root_path = os.path.abspath(path)
        self.image_train_path = os.path.join(self.root_path, 'images-train')
        self.image_test_path = os.path.join(self.root_path, 'images-test')
        self.label_train_path = os.path.join(self.root_path, 'train.jsonl')
        self.label_val_path = os.path.join(self.root_path, 'val.jsonl')
        self.label_test_path = os.path.join(self.root_path, 'test_cls.jsonl')

        self.train_gts = []
        self.val_gts = []
        self.test_gts = []

        with open(self.label_train_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.train_gts.append(json.loads(line))
        with open(self.label_val_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.val_gts.append(json.loads(line))
        with open(self.label_test_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.test_gts.append(json.loads(line))

        self.initialize()

    @staticmethod
    def poly2bbox(poly):
        key_points = list()
        rotated = collections.deque(poly)
        rotated.rotate(1)
        for (x0, y0), (x1, y1) in zip(poly, rotated):
            for ratio in (1 / 3, 2 / 3):
                key_points.append((x0 * ratio + x1 * (1 - ratio), y0 * ratio + y1 * (1 - ratio)))
        x, y = zip(*key_points)
        adjusted_bbox = (min(x), min(y), max(x) - min(x), max(y) - min(y))
        return key_points, adjusted_bbox

    def get_detection_gts(self):
        ground_truths = dict()

        total_train_gts = self.train_gts + self.val_gts
        for gt in tqdm(total_train_gts):
            width, height = gt['width'], gt['height']
            image_id = gt['image_id']
            image_name = gt['file_name']
            image_path = os.path.join(self.image_train_path, image_name)

            available_boxes = []

            annotations = gt['annotations']
            for sentence in annotations:
                sentence_chars = []
                polygons = []
                for char in sentence:
                    text = char['text']
                    polygon = char['polygon']
                    sentence_chars.append(text)
                    polygons.extend(polygon)

                points = minimal_rectangle(polygons, ordered=True).tolist()
                # key_points, adjusted_bbox = self.poly2bbox(polygons)
                # points = trans_xywh2corners(adjusted_bbox)
                available_boxes.append({
                    'points': points,
                    'transcription': ''.join(sentence_chars),
                })

            ignores = gt['ignore']
            for box in ignores:
                points = trans_integer(box['polygon'], width=width, height=height)
                available_boxes.append({
                    'points': points,
                    'transcription': '###',
                })

            ground_truths[image_path] = available_boxes

        return ground_truths


class BaiduCHSTRManager(DataManager):
    def __init__(self, path):
        super(BaiduCHSTRManager, self).__init__(
            name='baidu_ch_str',
            is_detection=False,
            is_recognition=True,
            is_scatter=False,
        )

        self.root_path = path
        self.image_train_path = os.path.join(self.root_path, 'train_images')

        with open(os.path.join(self.root_path, 'train.list'), 'r', encoding='utf-8') as f:
            self.lines = f.read().split('\n')

        self.initialize()

    def get_recognition_gts(self):
        ground_truths = dict()

        for line in tqdm(self.lines):
            width, height, image_name, text = line.split('\t')
            width, height = int(width), int(height)

            image_path = os.path.abspath(os.path.join(self.image_train_path, image_name))

            ground_truths[image_path] = {
                'text': text,
                'width': width,
                'height': height,
            }

        return ground_truths

    def recognition_output_paddle(self, output_dir, image_dir):
        gts = self.rec_gts or self.get_recognition_gts()
        lines = []
        for path, box in tqdm(gts.items()):
            text = box['text']
            line = '{}\t{}'.format(path, text)
            lines.append(line)

        with open(output_dir, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def save_cropped_image(self, output_dir, v2h=True, v2h_threshold=1.5):
        pass


if __name__ == '__main__':
    manager = BaiduCHSTRManager('../../data/Baidu-Chinese-Scene-Recog')
