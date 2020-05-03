To create some of the visualizations seen in the paper, you can try running driver2.py and driver3.py. driver.py offers some further visualizations that we did not use, and was largely a proof of concept. 

To run the files, you will have to install TensorFlow and Tensorflow Hub on the system, as well as MatPlotLib and cv2 for face detection. 

style_image  = ImageUtils.image_op(
    images = [content_image, style_image],
    # Change this from a + b to something else 
    # May use CNN to map features 
    # op     = lambda a, b: ImageUtils.clip_0_1(a*0.7 + b*1) # Overemphasizes style image
    op     = lambda a, b: ImageUtils.clip_0_1(a + b)
)

The above line can be modified to change the weighting of the new style image. It is present in both of the files. 

ImageUtils.image_op(
    images = [content_image, stylized_image, style_image],
    op     = lambda a, b, c: (a + b)/2
)

The above line or similar can be used to modify the weighting of the output image relative to the content, stylized, and style image. 

ImageUtils is a custom implementation and should be kept in the directory. Anime-images is an assumed directory for style repository images. The content images can be updated via URL in the files, and is pre-set to defaults. 