# Import os for traversing directory
import os

# Importing TensorFlow and 
# TensorFlow Hub for style transfer model
import tensorflow as tf
import tensorflow_hub as hub

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

import PIL
from PIL import Image

# Anime Style Images Dataset (Absolute and relative paths)
# https://github.com/Mckinsey666/Anime-Face-Dataset
# path_to_pics = "../../cropped/"
path_to_pics = "../Anime-Face-Dataset/cropped/"

# Grab content image from link
content_image = ImageUtils.grab_image('https://gradschool.cornell.edu/wp-content/uploads/2018/07/JonPark.jpg')

# Make list of style images and shuffle the list
dirs = os.listdir(path_to_pics)
shuffle(dirs)

# Opens MatPlotLib figure
fig = plt.figure(figsize=(12,6))
fig = plt.gcf()
fig.canvas.set_window_title('Style Transfer')

# For a whole directory
for i, filename in enumerate(dirs):

    # Limit number of style transfers
    if i > 5: break

    # Attempt style transfer on content image with current style image
    try: 
        style_image   = ImageUtils.grab_image(path_to_pics + filename)
        style_orig    = style_image
        style_image   = ImageUtils.clip_0_1(content_image + style_image)

        # Uses model from Tensor Flow Hub 
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

        # Uncomment this and comment out previous stylized_image definition to
        # map style image on content image without style transfer  
        # stylized_image = style_image

        # Clear figure and update images
        fig.clf()
        plt.subplot(1, 3, 1); ImageUtils.imshow(content_image,  'Content Image')
        plt.subplot(1, 3, 2); ImageUtils.imshow(stylized_image, 'New Image')
        plt.subplot(1, 3, 3); ImageUtils.imshow(style_orig,    'Style Image')
        ImageUtils.flashplot()

    except Exception as e:
        if input(f" - Image failed on {filename}: {e}.\n - Type q to quit or anything else to continue: ") == "q":
            break
input('Finish? : ')
