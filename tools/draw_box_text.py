# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : draw_box_text.py
@Date    : 2022/11/4 22:37 
"""

import os
import sys
import math
import json
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from constant import PATH_FONT, PATH_RESULT
from utils.draw import draw_ocr_box, draw_ocr_box_text

PATH_OUTPUT = os.path.join(PATH_RESULT, 'draw/dataset')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, help='path of text detection dataset ground truth file')
parser.add_argument('--dataset_name', type=str, help='name of dataset')
parser.add_argument('--output_dir', type=str, help='path to saving annotated images')
parser.add_argument('--font_path', default=PATH_FONT, type=str, help='file path of chinese font using in the image')
parser.add_argument('--draw_content', type=str, default='boxtext', help='`boxtext` or `box`')
args = parser.parse_args()


if __name__ == '__main__':
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    draw_content = args.draw_content
    output_dir = args.output_dir or os.path.join(os.path.join(PATH_OUTPUT, dataset_name), draw_content)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(dataset_dir, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in tqdm(lines):
        image_path, gts = line.split('\t')
        image_name = os.path.split(image_path)[1]
        gts = json.loads(gts)

        raw_image = Image.open(image_path)
        b_boxes = []
        for box in gts:
            b_boxes.append([tuple(point) for point in box['points']])

        if 'text' in draw_content:
            b_txts = []
            for box in gts:
                b_txts.append(box['transcription'])

        output_path = os.path.join(output_dir, image_name)
        if draw_content == 'box':
            output_image = draw_ocr_box(image=raw_image, boxes=b_boxes)
        else:
            output_image = draw_ocr_box_text(image=raw_image, boxes=b_boxes, txts=b_txts, font_path=args.font_path)
        output_image.save(output_path)
