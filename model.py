import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization
from decoderBlock import DecoderBlock

class GPT2Model(Model):
    def __init__(self, hparams):
        super(GPT2Model, self).__init__(name='GPT2Model')

        self.positional_encoding_weights = tf.Variable(tf.random.normal((hparams.context_length[0],
                                                                        hparams.embeding_size[0]),
                                                                        stddev=0.01,
                                                                        name= 'pos_enc'))

        self.word_embedding_weights = tf.Variable(tf.random.normal((hparams.vocab_size[0],
                                                                    hparams.embeding_size[0]),
                                                                    stddev=0.02, name = 'word_emb'))

        self.decoder_blocks = self.initDecoderBlocks(hparams)
        self.norm_layer = LayerNormalization()
        self.hparams = hparams


    def initDecoderBlocks(self, hparams):
        return [DecoderBlock(hparams) for _ in range(hparams.n_layer)]

    def expand_tile(self, value, size):
        """Add a new axis of given size."""
        value = tf.convert_to_tensor(value, name='value')
        ndims = value.shape.ndims
        return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)

    def positions_for(self, tokens, past_length):
        batch_size = tf.shape(tokens)[0]
        nsteps = tf.shape(tokens)[1]
        return self.expand_tile(past_length + tf.range(nsteps), batch_size)

    def call(self, input, **kwargs):
        x = input
        batch, sequence = [ob.value for ob in x.shape.dims]#shape_list(x)

        past = kwargs['past']
        past_length = 0 if past is None else tf.shape(past)[-2]

        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
        h = tf.gather(self.word_embedding_weights, x) + \
            tf.gather(self.positional_encoding_weights, self.positions_for(x, past_length))

        assert len(pasts) == self.hparams.n_layer

        for layerIndex, past in enumerate(pasts):
            h, present = self.decoder_blocks[layerIndex](h, past = past)
            presents.append(present)

        h = self.norm_layer(h)

        results = {}
        results['present'] = tf.stack(presents, axis=1)

        h_flat = tf.reshape(h, [batch * sequence, self.hparams.embeding_size[0]])
        logits = tf.matmul(h_flat, self.word_embedding_weights, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, self.hparams.vocab_size[0]])

        results['logits'] = logits
        return results