# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : dataset_generate_ctw.py
@Date    : 2022/11/5 0:02 
"""

import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from utils.data.manager import CTWManager
from constant import PATH_DATA, PATH_DATA_DET, PATH_DATA_RECOG, PATH_DATA_RECOG_IMG

PATH_LSVT = os.path.join(PATH_DATA, 'CTW')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default=PATH_LSVT, type=str, help='root path of ReCTS dataset files')
parser.add_argument('--det_dir', default=PATH_DATA_DET, type=str, help='path to save detection ground truth file')
parser.add_argument('--det_name', default='ctw.txt', type=str, help='file name of detection ground truth file')
parser.add_argument('--reg_dir', default=PATH_DATA_RECOG, type=str, help='path to save recognition ground truth files')
parser.add_argument('--reg_name', default='ctw.txt', type=str, help='file name of recognition ground truth file')
parser.add_argument('--no_crop', action='store_true', help='whether crop raw image into text box images for recognition')
parser.add_argument('--reg_img_dir', default=PATH_DATA_RECOG_IMG, type=str, help='path to save recognition images')
parser.add_argument('--rotate', action='store_false', help='whether rotate vertical image to horizontal')
parser.add_argument('--v2h_threshold', default=1.5, type=float, help='threshold of height/weight ratio deciding whether to rotate an image')
parser.add_argument('--direction', default='all', type=str, help='filter images by camera angle. must be one of (`all`, `side`)')
args = parser.parse_args()

if __name__ == '__main__':
    manager = CTWManager(path=args.dataset_dir, direction=args.direction)
    manager.detection_output_paddle(output_dir=os.path.join(args.det_dir, args.det_name))
    manager.recognition_output_paddle(output_dir=os.path.join(args.reg_dir, args.reg_name), image_dir=args.reg_img_dir)
    if not args.no_crop:
        manager.save_cropped_image(output_dir=args.reg_img_dir, v2h=args.rotate, v2h_threshold=args.v2h_threshold)
