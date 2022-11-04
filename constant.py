# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : constant.py
@Date    : 2022/9/13 23:59 
"""

import os

PATH_ROOT = os.path.abspath(os.path.split(os.path.abspath(__file__))[0])
PATH_DATA = os.path.join(PATH_ROOT, 'data')
PATH_RESULT = os.path.join(PATH_ROOT, 'results')

if not os.path.exists(PATH_RESULT):
    os.makedirs(PATH_RESULT)

# dataset
PATH_DATA_DET = os.path.join(PATH_DATA, 'det')
PATH_DATA_RECOG = os.path.join(PATH_DATA, 'recog')
PATH_DATA_RECOG_IMG = os.path.join(PATH_DATA_RECOG, 'image')

if not os.path.exists(PATH_DATA_DET):
    os.makedirs(PATH_DATA_DET)
if not os.path.exists(PATH_DATA_RECOG_IMG):
    os.makedirs(PATH_DATA_RECOG_IMG)

# PaddleOCR
PATH_PADDLEOCR = os.path.join(PATH_ROOT, 'PaddleOCR')
PATH_FONT = os.path.join(PATH_PADDLEOCR, 'doc/fonts/simfang.ttf')
