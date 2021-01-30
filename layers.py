import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from utils import shape_list

class Convolution1D(tf.keras.layers.Layer):
    def __init__(self, nx, nf, w_init_stdev = 0.02):
        super(Convolution1D, self).__init__()
        #self.w = tf.Variable(tf.random.normal((1, nx, nf),stddev=w_init_stdev),name='conv_1d/w')
        #self.b = tf.Variable(tf.constant(0.0, shape = [nf], dtype=tf.float32), name = 'conv_1d/b')
        self.nf = nf
        self.nx = nx
        self.w_init_stdev = w_init_stdev

    def build(self, input_shape):
        self.w = self.add_weight(
            "cov1d_weight",
            shape = [1, self.nx, self.nf],
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(
            stddev=self.w_init_stdev,
            mean=0.0),trainable=True)
        self.b = self.add_weight("conv1d_bias",
                                    shape=[self.nf],
                                    initializer=tf.constant_initializer(0.0),trainable=True)
        super(Convolution1D, self).build(input_shape)

    def call(self, input, **kwargs):
        x = input
        *start, nx = shape_list(x)
        newshape = start
        newshape.append(self.nf)
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, self.nx]),
                                 tf.reshape(self.w, [-1, self.nf])) + self.b, newshape)#start + [self.nf])
        return c

class MLP(tf.keras.layers.Layer):
    def __init__(self, n_state, nx = 768):
        super(MLP, self).__init__()
        self.conv_1 = Convolution1D(nx = nx, nf = n_state)
        self.conv_2 = Convolution1D(nx = n_state, nf = nx)

    def gelu(self, x):
        return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

    def call(self, input, **kwargs):
        x = input
        x = self.conv_1(x)
        x = self.gelu(x)
        x = self.conv_2(x)
        return x


class Attention(tf.keras.layers.Layer):
    def __init__(self, n_head, nx = 768,n_state = 768):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.conv_1 = Convolution1D(nx = nx, nf = n_state*3)
        self.conv_2 = Convolution1D(nx = nx, nf = n_state)

    def split_states(self, x, n):
        """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
        *start, m = shape_list(x)
        return tf.reshape(x, start + [n, m // n])

    def split_heads(self, x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(self.split_states(x, self.n_head), [0, 2, 1, 3])

    def attention_mask(self, nd, ns, *, dtype):
        """1's in the lower triangle, counting from the lower right corner.

        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)


    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = self.attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(self, q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))
        w = self.mask_attn_weights(w)
        w = K.softmax(w)
        a = tf.matmul(w, v)
        return a

    def merge_states(self, x):
        """Smash the last two dimensions of x into a single dimension."""
        *start, a, b = shape_list(x)
        return tf.reshape(x, start + [a * b])


    def merge_heads(self, x):
        # Reverse of split_heads
        return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def call(self, x, **kwargs):

        past = kwargs['past']
        x = self.conv_1(x)
        q, k, v = map(self.split_heads, tf.split(x, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = self.multihead_attn(q, k, v)
        a = self.merge_heads(a)
        a = self.conv_2(a)
        return a, present

