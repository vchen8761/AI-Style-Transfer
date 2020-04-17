class ModelLoader():

    vgg = tf.keras.applications.VGG19(include_top = True, weights = 'imagenet')
    content_layers = ['block5_conv2'] 
    style_layers   = [f'block{i}_conv1' for i in range(1,6)]


    def get_vgg_model(self, layers):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        outputs = [self.vgg.get_layer(name).output for name in layers]
        model = tf.keras.Model([this.vgg.input], outputs)
        return model    


    def get_layer_stats(self, image, layers):
        """Print statistics of each layer's output"""
        style_extractor = self.get_vgg_model(layers)
        style_outputs   = style_extractor(image * 255)

        #Look at the statistics of each layer's output
        for name, output in zip(layers, style_outputs):
            print(name)
            print("  shape : ", output.numpy().shape)
            print("  min   : ", output.numpy().min())
            print("  max   : ", output.numpy().max())
            print("  mean  : ", output.numpy().mean())
            print()



class StyleContentModel(tf.keras.models.Model):
    
    def __init__(self, vgg, style_layers, content_layers):
        
        super(StyleContentModel, self).__init__()
        
        self.model            = vgg.get_vgg_model(style_layers + content_layers)
        
        self.style_layers     = style_layers
        self.content_layers   = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable    = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        
        # Scale back the pixel values
        inputs *= 255.0
        
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