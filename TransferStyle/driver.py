# Importing TensorFlow
import sys

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

from StyleTrans import *
from ImageUtils import *
from StyleContentModel import *

plt.figure(figsize=(12,6))

ImageUtils.enableflashplot()

transferer = StyleTrans(
    content_path = 'https://gradschool.cornell.edu/wp-content/uploads/2018/07/JonPark.jpg',
    style_path   = 'https://image.freepik.com/free-vector/abstract-dynamic-pattern-wallpaper-vector_53876-59131.jpg' 
    #'style_path   = images/deku.jpg'
)

# # Averaging two images
# images = (transferer.content_image, transferer.style_image)
# transferer.content_image = (images[0] + images[1])/2
# transferer.style_image = images[0]
# transferer.updateImage()

def per_step(img, e, s):
    plt.subplot(1, 3, 1); ImageUtils.imshow(transferer.content_image,   'Content Image')
    plt.subplot(1, 3, 2); ImageUtils.imshow(img, f'New Image, epoch {e}, step {s}')
    plt.subplot(1, 3, 3); ImageUtils.imshow(transferer.style_image,     'Style Image')
    ImageUtils.flashplot()
    fig = plt.figure(1) 
    fig.clf() 
    ax = fig.subplots(nrows=2, ncols=1)

print("Starting Run")

image = transferer.run(epochs = 4, steps = 4, per_step = per_step)

plt.subplot(1, 3, 1); ImageUtils.imshow(transferer.content_image,   'Content Image')
plt.subplot(1, 3, 2); ImageUtils.imshow(image,                      'New Image')
plt.subplot(1, 3, 3); ImageUtils.imshow(transferer.style_image,     'Style Image')
plt.show()

input("Proceed: ")


# The following shows how the high frequency components have increased.
# Also, this high frequency component is basically an edge-detector. 

def high_pass_x_y(image):
    """For visualizing produced artifacts"""
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var

x_deltas, y_deltas = high_pass_x_y(transferer.content_image)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
ImageUtils.imshow(ImageUtils.clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

plt.subplot(2,2,2)
ImageUtils.imshow(ImageUtils.clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2,2,3)
ImageUtils.imshow(ImageUtils.clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

plt.subplot(2,2,4)
ImageUtils.imshow(ImageUtils.clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")
plt.show()

input("Conclude: ")

tf.image.total_variation(image).numpy()

file_name = 'output.png'
ImageUtils.tensor_to_image(image).save(file_name)

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(file_name)