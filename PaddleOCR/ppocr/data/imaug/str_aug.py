# -*- coding: utf-8 -*-

"""
@Author  : pixelock
@File    : str_aug.py
@Date    : 2023/2/21 21:14

Code: https://github.com/roatienza/straug
Paper: https://arxiv.org/abs/2108.06949
"""


import numpy as np
from PIL import Image
from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.warp import Curve, Distort, Stretch
from straug.weather import Fog, Snow, Frost, Rain, Shadow


class STRAug(object):
    def __init__(self, n=3, prob=0.5, mag_min=-1, mag_max=3, **kwargs):
        rng = np.random.default_rng(2023)
        self.n = n
        self.prob = prob
        self.mag_min = mag_min
        self.mag_max = mag_max

        ops = []
        ops.extend([Curve(rng=rng), Distort(rng), Stretch(rng)])  # warp
        ops.extend([Perspective(rng), Rotate(rng=rng), Shrink(rng), TranslateX(rng), TranslateY(rng)])  # geometry
        ops.extend([Grid(rng), VGrid(rng), HGrid(rng), RectGrid(rng), EllipseGrid(rng)])  # create different grids
        ops.extend([GaussianBlur(rng), DefocusBlur(rng), MotionBlur(rng), GlassBlur(rng), ZoomBlur(rng)])  # generate synthetic blur
        ops.extend([GaussianNoise(rng), ShotNoise(rng), ImpulseNoise(rng), SpeckleNoise(rng)])  # add noise
        ops.extend([Fog(rng), Snow(rng), Frost(rng), Rain(rng), Shadow(rng)])  # simulate certain weather conditions
        ops.extend([Contrast(rng), Brightness(rng), JpegCompression(rng), Pixelate(rng)])  # simulate camera sensor tuning and image compression/resizing
        ops.extend([Posterize(rng), Solarize(rng), Invert(rng), Equalize(rng), AutoContrast(rng), Sharpness(rng), Color(rng)])  # all other image processing issues
        ops.extend([])

        self.ops = ops

    def __call__(self, data):
        image_data = data['image']
        image = Image.fromarray(image_data)

        augments = np.random.choice(self.ops, self.n)
        for op in augments:
            image = op(image, mag=np.random.randint(self.mag_min, self.mag_max), prob=self.prob)

        image_data = np.asarray(image)
        data['image'] = image_data
        return data
