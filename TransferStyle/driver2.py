# Importing TensorFlow
import sys, os

import tensorflow as tf

# For plotting
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

# Other helpers
import numpy as np
import PIL.Image
import time
import functools
from random import shuffle

from StyleTrans import *
from ImageUtils import *
from StyleContentModel import *

import tensorflow_hub as hub

import PIL
from PIL import Image

# https://github.com/Mckinsey666/Anime-Face-Dataset
# path_to_pics = "../../cropped/"
path_to_pics = "../Anime-Face-Dataset/cropped/"

content_image = ImageUtils.grab_image('https://gradschool.cornell.edu/wp-content/uploads/2018/07/JonPark.jpg')

dirs = os.listdir(path_to_pics)
shuffle(dirs)

ImageUtils.enableflashplot()
fig = plt.figure(figsize=(12,6))
# For a whole directory
for i, filename in enumerate(dirs):
    if i > 5: break
    try: 
        style_image   = ImageUtils.grab_image(path_to_pics + filename)
        style_orig    = style_image

        style_image = ImageUtils.clip_0_1(content_image + style_image)

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        #stylized_image = style_image

        fig.clf()
        plt.subplot(1, 3, 1); ImageUtils.imshow(content_image,  'Content Image')
        plt.subplot(1, 3, 2); ImageUtils.imshow(stylized_image, 'New Image')
        plt.subplot(1, 3, 3); ImageUtils.imshow(style_orig,    'Style Image')
        ImageUtils.flashplot()

    except Exception as e:
        if input(f" - Image failed on {filename}: {e}.\n - Type q to quit or anything else to continue: ") == "q":
            break
input('Finish? : ')
