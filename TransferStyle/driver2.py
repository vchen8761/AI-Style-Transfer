# Importing TensorFlow and 
# TensorFlow Hub for style transfer model
print(" - Starting to import TensorFlow")
import tensorflow as tf
print("\t> Finished")
import tensorflow_hub as hub

# Import os for traversing directory
from os import listdir
from sys import platform

# For plotting
import matplotlib.pyplot as plt

# Other helpers
from random import shuffle

from StyleContentModel  import StyleContentModel
from ImageUtils         import ImageUtils
from StyleTrans         import StyleTrans

# Anime Style Images Dataset (Absolute and relative paths)
# https://github.com/Mckinsey666/Anime-Face-Dataset
if platform == "darwin":
    # Specifics for Vadim
    path_to_pics = "../../cropped/"

else:
    # Specifics for Victor
    path_to_pics = "../Anime-Face-Dataset/cropped/"

# Grab content image from link
content_image = ImageUtils.grab_image('https://gradschool.cornell.edu/wp-content/uploads/2018/07/JonPark.jpg')
# content_image = ImageUtils.grab_image('images/cloud1.jpg')

# Uses model from Tensor Flow Hub 

print(" - Loading pre-trained model from hub")
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
print("\t> Finished")

# Make list of style images and shuffle the list
dirs = listdir(path_to_pics)
shuffle(dirs)

# Opens MatPlotLib figure
fig = plt.figure(figsize=(12,6))
fig = plt.gcf()
fig.canvas.set_window_title('Style Transfer')

# For a whole directory
fails = 0
for i, filename in enumerate(dirs):

    # Limit number of style transfers
    if i - fails > 10: break

    # Attempt style transfer on content image with current style image
    try: 
        style_image  = ImageUtils.grab_image(path_to_pics + filename)
        
        # Testing code begins here

        # Attempted to use PIL image resizing
        # style_image = Image.open(path_to_pics + filename)
        # style_image = style_image.resize((1000,1000))

        # Attempting to resize style image for hypothesis
        # So far just makes style image really pixelated
        style_image = tf.image.resize(style_image, (15,15),
                preserve_aspect_ratio = True, antialias = True)

        # Used practice from Deep Dream Tensorflow tutorial - did not work :(
        # style_image = tf.keras.applications.inception_v3.preprocess_input(style_image)
        # style_image = tf.convert_to_tensor(style_image)

        # Scales image to have mean 0 and variance 1
        # Oh no ... His power level is over 9000!
        # style_image = tf.image.per_image_standardization(style_image)

        # Random saturation does not do much ... just looks like a cursed
        # heatmap
        # style_image = tf.image.random_saturation(style_image, 5, 10)

        # Testing code ends here

        style_orig   = style_image
        style_image  = ImageUtils.image_op(
            images = [content_image, style_image],
            op     = lambda a, b: ImageUtils.clip_0_1(a + b)
        )

        print(" - Generating image", i)
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

        # Uncomment this and comment out previous stylized_image definition to
        # map style image on content image without style transfer  
        # stylized_image = style_image

        # Clear figure and update images
        fig.clf()
        plotset1 = (    # This shows 2x3 5-plot layout
            ((2,3,1), content_image,    'Content Image'),
            ((2,3,4), stylized_image,       'New Image'),

            ((1,3,2), ImageUtils.image_op(
                images = [content_image, stylized_image, style_image],
                op     = lambda a, b, c: (2 * a + 2 * b)/4
            ), 'Average Image'),

            ((2,3,3), style_orig,         'Style Image'),
            ((2,3,6), style_image,    'New Style Image'),
        )

        plotset2 = (    # This shows 1x3 simple layout
            ((1,3,1), content_image,    'Content Image'),
            ((1,3,2), stylized_image,       'New Image'),
            ((1,3,3), style_orig,         'Style Image'),
        )

        for c, i, t in plotset1:
            plt.subplot(*c)
            ImageUtils.imshow(i, t)
       
        # View style transfers one by one
        input('Next? : ')

        ImageUtils.flashplot()

        print("\t> Finished")

    except Exception as e:
        print(" - Failed on image", i)
        print(f"\t> Image failed on {filename}: {e}.")
        fails += 1      # This skips fails. To be fixed later

        # For testing
        if fails > 5: quit()
        
        # if "n" == input(f" - Image failed on {filename}: {e}.\n - Type 'n' if to stop, anything else to go: "):
            # break


input('Finish? : ')
