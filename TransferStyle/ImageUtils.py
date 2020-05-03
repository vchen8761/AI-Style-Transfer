# Importing TensorFlow
import tensorflow as tf

# For plotting
import matplotlib.pyplot as plt

# Other helpers
import numpy as np

class ImageUtils:

    def tensor_to_image(tensor):
        """Pulls image from tensor"""
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)


    def load_img(path_to_img, max_dim = 512):
        """
        Function to load an image through TF and limit its maximum dimension
        """

        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels = 3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        scale = max_dim / max(shape)

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def image_op(images, op, dim_of = 0):
        shapes = [
            tf.cast(tf.shape(i)[1:3], tf.float32) 
            for i in images
        ]

        for i in range(len(images)):
            if any(shapes[i] != shapes[dim_of]):
                images[i] = tf.image.resize(
                    images[i], 
                    tf.cast(shapes[dim_of], tf.int32))
        
        return op(*images)

    def grab_image(path):
        if len(path) > 5 and path[0:4] in ("www.", "http"):
            return ImageUtils.load_img(tf.keras.utils.get_file(
                fname = path.replace("/", "").replace("\\", ""), origin = path))
        else: return ImageUtils.load_img(path)


    def clip_0_1(image):
        """A function to keep the pixel values between 0 and 1"""
        return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)
        

    def imshow(image, title=None):
        """A simple function to display an image"""
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)

    def flashplot():
        plt.draw()
        plt.pause(0.001)

    def enableflashplot():
        plt.ion()
        plt.show()
        plt.pause(0.01)
