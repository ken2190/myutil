from keras import backend as K
from keras.initializers import glorot_normal
from keras.layers import Layer


class GaussianLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        """ GaussianLayer:__init__.
        Doc::
                
                    Args:
                        output_dim:     
                        **kwargs:     
                    Returns:
                       
        """
        self.output_dim = output_dim
        self.kernel_1, self.kernel_2, self.bias_1, self.bias_2 = [], [], [], []
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """ GaussianLayer:build.
        Doc::
                
                    Args:
                        input_shape:     
                    Returns:
                       
        """
        n_weight_rows = input_shape[2]
        self.kernel_1 = self.add_weight(name='kernel_1',
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                      shape=(self.output_dim,),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                      shape=(self.output_dim,),
                                      initializer=glorot_normal(),
                                      trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        """ GaussianLayer:call.
        Doc::
                
                    Args:
                        x:     
                    Returns:
                       
        """
        output_mu = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        return [output_mu, output_sig_pos]

    def compute_output_shape(self, input_shape):
        """.
        Doc::
                
                    The assumption is the output ts is always one-dimensional
        """
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]
