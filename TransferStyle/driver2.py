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
path_to_pics = "../../cropped/"

content_image = ImageUtils.grab_image('https://gradschool.cornell.edu/wp-content/uploads/2018/07/JonPark.jpg')

dirs = os.listdir(path_to_pics)
shuffle(dirs)

# For a whole directory
for filename in dirs:
    try: 
        style_image   = ImageUtils.grab_image(path_to_pics + filename)
        style_orig    = style_image

        clip = ImageUtils.clip_0_1

        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

        plt.figure(figsize=(12,6))
        ImageUtils.enableflashplot()
        plt.subplot(1, 3, 1); ImageUtils.imshow(content_image,  'Content Image')
        plt.subplot(1, 3, 2); ImageUtils.imshow(stylized_image, 'New Image')
        plt.subplot(1, 3, 3); ImageUtils.imshow(style_image,    'Style Image')
        ImageUtils.flashplot()

        fig = plt.figure(1) 
        fig.clf() 
        ax = fig.subplots(nrows=2, ncols=1)

    except Exception as e:
        if input(f" - Image failed on {filename}: {e}.\n - Type q to quit or anything else to continue: ") == "q":
            break