# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : dataset_generate_baidu_chstr.py
@Date    : 2022/11/25 0:25 
"""

import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from utils.data.manager import BaiduCHSTRManager
from constant import PATH_DATA, PATH_DATA_DET, PATH_DATA_RECOG, PATH_DATA_RECOG_IMG

PATH_LSVT = os.path.join(PATH_DATA, 'Baidu-Chinese-Scene-Recog')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default=PATH_LSVT, type=str, help='root path of Baidu Chinese Scene Recog dataset files')
parser.add_argument('--reg_dir', default=PATH_DATA_RECOG, type=str, help='path to save recognition ground truth files')
parser.add_argument('--reg_name', default='baidu_chstr.txt', type=str, help='file name of recognition ground truth file')  # baidu_chstr_tidy.txt
parser.add_argument('--save_image', action='store_false', help='whether apply rotate and save rotated images')
parser.add_argument('--reg_img_dir', default=PATH_DATA_RECOG_IMG, type=str, help='path to save recognition images')
parser.add_argument('--tidy', action='store_true', help='whether apply data cleaning and clustering')
parser.add_argument('--rotate', action='store_false', help='whether rotate vertical image to horizontal')
parser.add_argument('--v2h_threshold', default=1.5, type=float, help='threshold of height/weight ratio deciding whether to rotate an image')
args = parser.parse_args()

if __name__ == '__main__':
    manager = BaiduCHSTRManager(path=args.dataset_dir, tidy=args.tidy)
    manager.recognition_output_paddle(output_dir=os.path.join(args.reg_dir, args.reg_name), image_dir=args.reg_img_dir)
    if args.save_image:
        manager.save_cropped_image(output_dir=args.reg_img_dir, v2h=args.rotate, v2h_threshold=args.v2h_threshold)
