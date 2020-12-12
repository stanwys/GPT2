from tensorflow.keras.layers import LayerNormalization
from layers import Attention, MLP
import tensorflow as tf

class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, hparams):
        super(DecoderBlock, self).__init__(name='DecoderBlock')

        self.attention_layer = Attention(n_head=hparams.n_head[0],
                                         nx = hparams.embeding_size[0],
                                         n_state= hparams.embeding_size[0])
        self.mlp_layer = MLP(n_state= 4 * hparams.embeding_size[0])
        self.norm_layer1 = LayerNormalization()
        self.norm_layer2 = LayerNormalization()

    def call(self, inputs, **kwargs):
        x = inputs
        x_norm_1 = self.norm_layer1(x)
        a, present = self.attention_layer(input = x_norm_1, past = kwargs['past'])
        x = x + a
        x_norm_2 = self.norm_layer2(x)
        m = self.mlp_layer(x_norm_2)
        x = x + m
        return x, present