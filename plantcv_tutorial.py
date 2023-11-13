# Import libraries
import os
import numpy as np
import cv2
from plantcv import plantcv as pcv
pcv.params.debug = "plot"
# pip install plantcv
# pip install plantcv jupyterlab ipympl

# load image
img, path, filename = pcv.readimage(filename="img/original_image.jpg")

s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, object_type='light')

s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=3)