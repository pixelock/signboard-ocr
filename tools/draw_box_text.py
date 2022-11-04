# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : draw_box_text.py
@Date    : 2022/11/4 22:37 
"""

import os
import math
import json
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from constant import PATH_FONT, PATH_RESULT

PATH_OUTPUT = os.path.join(PATH_RESULT, 'draw/dataset')


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path=PATH_FONT):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return img_show


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, help='path of text detection dataset ground truth file')
parser.add_argument('--dataset_name', type=str, help='name of dataset')
parser.add_argument('--font_path', default=PATH_FONT, type=str, help='file path of chinese font using in the image')
parser.add_argument('--output_dir', type=str, help='path to saving annotated images')
args = parser.parse_args()


if __name__ == '__main__':
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    output_dir = args.output_dir or os.path.join(PATH_OUTPUT, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(dataset_dir, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in tqdm(lines):
        image_path, gts = line.split('\t')
        image_name = os.path.split(image_path)[1]
        gts = json.loads(gts)

        raw_image = Image.open(image_path)
        b_boxes, b_txts = [], []
        for box in gts:
            b_boxes.append([tuple(point) for point in box['points']])
            b_txts.append(box['transcription'])

        output_path = os.path.join(output_dir, image_name)
        output_image = draw_ocr_box_txt(image=raw_image, boxes=b_boxes, txts=b_txts, font_path=args.font_path)
        output_image.save(output_path)
