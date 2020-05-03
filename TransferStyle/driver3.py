# Importing TensorFlow and 
# TensorFlow Hub for style transfer model
print(" - Starting to import TensorFlow")
import tensorflow as tf
print("\t> Finished")
import tensorflow_hub as hub

# Import os for traversing directory
from os import listdir

import sys
from sys import platform

# For plotting
import matplotlib.pyplot as plt

# Other helpers
from random import shuffle

from StyleContentModel  import StyleContentModel
from ImageUtils         import ImageUtils
from StyleTrans         import StyleTrans

# For removing background image/flood fill
# sys.path.insert(1, './image-background-removal/')
# import seg

# For testing face detection using OpenCV
import cv2
import os.path

# Anime Style Images Dataset (Absolute and relative paths)
# https://github.com/Mckinsey666/Anime-Face-Dataset
# if platform == "darwin":
#     # Specifics for Vadim
#     path_to_pics = "../../cropped/"

# else:
    # Specifics for Victor
    # path_to_pics = "../Anime-Face-Dataset/cropped/"
path_to_pics = "./anime-images/"

# Grab content image from link
# content_image = ImageUtils.grab_image('https://gradschool.cornell.edu/wp-content/uploads/2018/07/JonPark.jpg')
# content_image = ImageUtils.grab_image('http://www.mathcs.richmond.edu/~jdenny/Jory.jpg')
content_image = ImageUtils.grab_image('https://facultystaff.richmond.edu/~dszajda/images/doug_small_website_photo_UR_Fall_2011.jpg')

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


# Helper functions

# Use OpenCV to detect anime faces 
# Source: https://github.com/nagadomi/lbpcascade_animeface
def detectAnime(filename, index, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(
        gray,
        # detector options
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (24, 24)
    )

    # If face not detected then faces tuple is empty
    if not any(map(len, faces)):
        return
    else:
        x, y, w, h = faces[0]
        temp = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("AnimeFaceDetect", image)
        # cv2.waitKey(0)

        # Crop using numpy slicing and bounding box
        crop_img = image[y:y+h, x:x+w]

        cv2.imwrite("out.png", crop_img)
        return faces[0]

        # Use this to filter out original anime face dataset for detectable
        # faces with OpenCV method above
        # cv2.imwrite("./anime-images/out" + str(index) + ".png", image)


# For a whole directory
fails = 0
for i, filename in enumerate(dirs):

    # Limit number of style transfers
    if i - fails >= 5: break

    # Attempt style transfer on content image with current style image
    try: 
        # style_image  = ImageUtils.grab_image(path_to_pics + filename)
        
        # Testing code begins here

        # Trying to remove image background
        # seg.run_visualization(path_to_pics + filename)
        # style_image = ImageUtils.grab_image('test.jpg')

        # Attempting to resize style image for hypothesis
        # So far just makes style image really pixelated
        # style_image = tf.image.resize(
        #        style_image, (12,12), method='area',
        #        preserve_aspect_ratio = True, antialias = True)

        # Trying to match faces using resizing but need to have a CNN detect
        # facial features in images and map the style image onto content image 
        # style_image = tf.image.resize_with_crop_or_pad(style_image, 1500,1500)
       
        # Trying face detection with OpenCV
        x, y, w, h = detectAnime(path_to_pics + filename, i)
        if x is not None:
            print(x, y, w, h)
        style_image = ImageUtils.grab_image("out.png")
        # style_image = tf.image.resize_with_crop_or_pad(style_image, 1000,1000)

        # Testing code ends here

        style_orig   = style_image
        style_image  = ImageUtils.image_op(
            images = [content_image, style_image],
            # Change this from a + b to something else 
            # May use CNN to map features 
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
                op     = lambda a, b, c: (a + b)/2
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
