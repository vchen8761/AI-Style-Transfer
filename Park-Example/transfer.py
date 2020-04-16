# Importing TensorFlow
import sys

import tensorflow as tf

# For plotting
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,8)
mpl.rcParams['axes.grid'] = False

plt.figure(figsize=(12,6))

# Other helpers
import numpy as np
import PIL.Image
import time
import functools

from st_helpers import *

# Load image data
content_path = tf.keras.utils.get_file(
    fname  = 'park.jpg',
    origin = 'https://gradschool.cornell.edu/wp-content/uploads/2018/07/JonPark.jpg'
)

style_path = tf.keras.utils.get_file(
    fname  = 'YellowLabradorLooking_new.jpg', 
    origin = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
)

content_image = load_img(content_path)
style_image = load_img(style_path)

# Show images
plt.subplot(1, 2, 1); imshow(content_image, 'Content Image')
plt.subplot(1, 2, 2); imshow(style_image,   'Style Image')
plt.ion()
plt.show()
plt.pause(0.01)



print(" - Outputted image! ")



# Load a VGG19 and test run it on our image to ensure it's used correctly:
x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape

# Just test the model to make sure it's looking at the right thing
predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

# Now load a VGG19 without the classification head, and list the layer names
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# print()
# for layer in vgg.layers:
#     print(layer.name)
    
# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer of interest
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1', 
    'block4_conv1', 
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers   = len(style_layers)


# Create the model
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

# Extract style and content
extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']


print(" - Just finished model creation!")



# Run gradient descent
# Set your style and content target values:
style_targets   = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Define a tf.Variable to contain the image to optimize.
image = tf.Variable(content_image)

# Create an optimizer. The paper recommends LBFGS, but Adam works okay, too:
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# To optimize this, use a weighted combination of the two losses to get the total loss:
style_weight   = 1e-2
content_weight = 1e4



print(" - Just finished gradient descent setup! Starting to train!")


# Pulling in variables from scope
style_content_loss = lambda output: style_content_loss_template(
        output,
        style_targets,      style_weight,   num_style_layers,
        content_targets,    content_weight, num_content_layers, 
    )


import time
start = time.time()

epochs = 5          # Set low for our purposes
steps_per_epoch = 5 # Set low for our purposes

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image, extractor, style_content_loss, opt)
        print(".", end='')
        imshow(image, f'New Image, epoch {n}, step {m}')
        plt.draw()
        plt.pause(0.001)
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print("Train step: {}".format(step))
    
end = time.time()
print("Total time: {:.1f}".format(end-start))


print(" - Finished!")

input("Proceed to final steps? : ")



plt.rcParams['figure.figsize'] = [15, 10]

plt.subplot(1, 3, 1); imshow(content_image, 'Content Image')
plt.subplot(1, 3, 2); imshow(image, 'New Image')
plt.subplot(1, 3, 3); imshow(style_image,   'Style Image')
plt.show()


print(" - Just plotted results: ")


# The following shows how the high frequency components have increased.
# Also, this high frequency component is basically an edge-detector. 

x_deltas, y_deltas = high_pass_x_y(content_image)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

plt.subplot(2,2,2)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2,2,3)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

plt.subplot(2,2,4)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")

plt.show()






file_name = input("Save to a file of name: ")
tensor_to_image(image).save(file_name)

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(file_name)