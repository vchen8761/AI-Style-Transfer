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


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)



def load_img(path_to_img):
    """Function to load an image and limit its maximum dimension to 512 pixels."""
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img



def imshow(image, title=None):
    """A simple function to display an image"""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
        
        
def clip_0_1(image):
    """A function to keep the pixel values between 0 and 1"""
    return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)
        
        
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model



def gram_matrix(input_tensor):
    """Computes gram matrix for an input tensor"""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape   = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations



class StyleContentModel(tf.keras.models.Model):
    
    def __init__(self, style_layers, content_layers):
        
        super(StyleContentModel, self).__init__()
        
        self.vgg              = vgg_layers(style_layers + content_layers)
        
        self.style_layers     = style_layers
        self.content_layers   = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable    = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        
        # Scale back the pixel values
        inputs = inputs * 255.0
        
        # Preprocess them with respect to VGG19 stats
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        
        # Pass through the mini network
        outputs = self.vgg(preprocessed_input)
        
        # Segregate the style and content representations
        style_outputs, content_outputs = (
            outputs[:self.num_style_layers], 
            outputs[self.num_style_layers:]
        )

        # Calculate the gram matrix for each layer
        style_outputs = [
            gram_matrix(style_output) 
            for style_output in style_outputs
        ]

        # Assign the content representation and gram matrix in layer-by-layer fashion
        content_dict = {
            content_name:value 
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name:value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {'content' : content_dict, 'style' : style_dict}
    

def style_content_loss_template(outputs, 
    style_targets,      style_weight,   num_style_layers,
    content_targets,    content_weight, num_content_layers,    
):
    """Basic loss function for total loss"""
    
    style_outputs   = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = tf.add_n([
        tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
            for name in style_outputs.keys()
    ])
    
    style_loss *= style_weight / num_style_layers         # NORMALIZE STEP

    content_loss = tf.add_n([
        tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
            for name in content_outputs.keys()
    ])
    content_loss *= content_weight / num_content_layers   # NORMALIZE STEP
    loss = style_loss + content_loss                      # SUM UP LOSS
    
    return loss


def high_pass_x_y(image):
    """For visualizing produced artifacts"""
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var



def total_variation_loss(image):
    """The regularization loss associated with image is the sum of the squares of the values"""
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))



# Update include it in the train_step function:
@tf.function()
def train_step(image, extractor, style_content_loss, opt, total_variation_weight = 0):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))