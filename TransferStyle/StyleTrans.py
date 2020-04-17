from StyleContentModel import StyleContentModel
from ImageUtils import ImageUtils

# Importing TensorFlow
import tensorflow as tf

class StyleTrans:

    def __init__(self, content_path, style_path):
        """
        Default constructor
        """

        self.content_image = ImageUtils.grab_image(content_path)
        self.style_image   = ImageUtils.grab_image(style_path)

        """
        Load pre-trained VGG19 network architecture
        """
        self.vgg = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')

        # Content layer where we will pull our feature maps
        self.content_layers = ['block5_conv2'] 
        self.style_layers   = [f'block{i}_conv1' for i in range(1,6)]

        # Define a tf.Variable to contain the image to optimize.
        self.updateImage()


    def updateImage(self):
        """Update a tf.Variable to contain the image to optimize."""
        self.image = tf.Variable(self.content_image)


    def run(self, epochs = 2, steps = 2, plot = True, 
        per_epoch = lambda img, e, s: 0,
        per_step  = lambda img, e, s: 0
    ): 

        # Create the model
        style_extractor = self.get_mini_model(self.style_layers)
        style_outputs = style_extractor(self.style_image * 255)

        # Extract style and content
        extractor = StyleContentModel(
            style_layers   = self.style_layers, 
            content_layers = self.content_layers, 
            get_mini_model = lambda l : self.get_mini_model(l)
        )

        # Run gradient descent
        # Set your style and content target values:
        style_targets   = extractor(self.style_image)['style']
        content_targets = extractor(self.content_image)['content']

        # Create an optimizer. The paper recommends LBFGS, but Adam works okay, too:
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        # To optimize, use a weighted combination of the two losses to get the total loss:
        style_weight   = 1e-2
        content_weight = 1e4

        loss_function = lambda output: self.style_content_loss_template(
            output,
            style_targets,      style_weight,   len(self.style_layers),
            content_targets,    content_weight, len(self.content_layers), 
        )

        import time
        start = time.time()

        step = 0
        for n in range(epochs):
            for m in range(steps):
                step += 1
                self.train_step(self.image, extractor, loss_function, opt) # modifies image
                print(".", end='')
                per_step(self.image, n, m)
            
            print("Train step: {}".format(step))
            per_epoch(self.image, n, m)
            
        end = time.time()
        print("Total time: {:.1f}".format(end - start))

        return self.image


    @tf.function()
    def train_step(self,
        image,  extractor,  loss_function, 
        opt,    total_variation_weight = 0
    ):
        """
        Mutates image as a step to conform to training
        """
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = loss_function(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(ImageUtils.clip_0_1(image))


    ########################################################################
    ### MODEL LOADING FUNCTIONS
    ########################################################################


    def get_mini_model(self, layers):
        """
        Create a custom VGG model which will be composed of specified layers. 
        Help run forward passes on the images and extract the necessary features.
        """
        outputs = [self.vgg.get_layer(name).output for name in layers]
        model = tf.keras.Model([self.vgg.input], outputs)
        return model


    def get_layer_stats(self, layers, image):
        """
        Print statistics of each layer's output
        """
        style_extractor = self.get_vgg_model(layers)
        style_outputs   = style_extractor(image)

        #Look at the statistics of each layer's output
        for name, output in zip(layers, style_outputs):
            print(name)
            print("  shape : ", output.numpy().shape)
            print("  min   : ", output.numpy().min())
            print("  max   : ", output.numpy().max())
            print("  mean  : ", output.numpy().mean())
            print()
    

    ########################################################################
    ### MODEL LOADING FUNCTIONS
    ########################################################################

    def style_content_loss_template(self, outputs, 
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
