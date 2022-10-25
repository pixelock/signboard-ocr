# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : constant.py
@Date    : 2022/9/13 23:59 
"""

import os

PATH_ROOT = os.path.split(os.path.abspath(__file__))[0]
PATH_DATA = os.path.join(PATH_ROOT, 'data')

# dataset
PATH_DATA_DET = os.path.join(PATH_DATA, 'det')
PATH_DATA_RECOG = os.path.join(PATH_DATA, 'recog')
PATH_DATA_RECOG_IMG = os.path.join(PATH_DATA_RECOG, 'image')

if not os.path.exists(PATH_DATA_RECOG_IMG):
    os.makedirs(PATH_DATA_RECOG_IMG)
