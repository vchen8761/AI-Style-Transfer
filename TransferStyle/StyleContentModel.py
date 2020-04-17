import tensorflow as tf

class StyleContentModel(tf.keras.models.Model):
    
    """
    This will be used for returning the content and style 
    features from the respective images.
    """


    def __init__(self, style_layers, content_layers, get_mini_model):
        
        super(StyleContentModel, self).__init__()
        
        self.vgg              = get_mini_model(style_layers + content_layers)
        
        self.style_layers     = style_layers
        self.content_layers   = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable    = False


    def gram_matrix(self, input_tensor):
        """Computes gram matrix for an input tensor"""
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape   = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations


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
            self.gram_matrix(style_output) 
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