# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : dataset_generate_rects.py
@Date    : 2022/10/25 23:09 
"""

import os
import argparse

from utils.data.manager import ReCTSManager
from constant import PATH_DATA, PATH_DATA_DET, PATH_DATA_RECOG, PATH_DATA_RECOG_IMG

PATH_RECTS = os.path.join(PATH_DATA, 'ICDAR-2019-ReCTS')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default=PATH_RECTS, type=str, help='root path of ReCTS dataset files')
parser.add_argument('--det_dir', default=PATH_DATA_DET, type=str, help='path to save detection ground truth file')
parser.add_argument('--det_name', default='rects.txt', type=str, help='file name of detection ground truth file')
parser.add_argument('--reg_dir', default=PATH_DATA_RECOG, type=str, help='path to save recognition ground truth files')
parser.add_argument('--reg_name', default='rects.txt', type=str, help='file name of recognition ground truth file')
parser.add_argument('--reg_img_dir', default=PATH_DATA_RECOG_IMG, type=str, help='path to save recognition images')
parser.add_argument('--no_rotate', action='store_false', help='whether rotate vertical image to horizontal')
parser.add_argument('--v2h_threshold', default=1.5, type=float, help='threshold of height/weight ratio deciding whether to rotate an image')
args = parser.parse_args()

if __name__ == '__main__':
    manager = ReCTSManager(path=args.dataset_dir)
    manager.detection_output_paddle(output_dir=os.path.join(args.det_dir, args.det_name))
    manager.recognition_output_paddle(output_dir=os.path.join(args.reg_dir, args.reg_name), image_dir=args.reg_img_dir)
    manager.save_cropped_image(output_dir=args.reg_img_dir, v2h=args.no_rotate, v2h_threshold=args.v2h_threshold)
