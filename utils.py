import tensorflow as tf

class HParams():
    def __init__(self, n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12):
        self.vocab_size = n_vocab,
        self.context_length = n_ctx,
        self.embeding_size = n_embd,
        self.n_head = n_head,
        self.n_layer = n_layer

    def override_from_dict(self, jsonObject):
        self.vocab_size = jsonObject['n_vocab']
        self.context_length = jsonObject['n_ctx']
        self.embeding_size = jsonObject['n_embd']
        self.n_head = jsonObject['n_head']
        self.n_layer = jsonObject['n_layer']

def default_hparams():
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12
    )

def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head[0], sequence, hparams.embeding_size[0] // hparams.n_head[0]]

def top_k_logits(logits, k):
    if k == 0:
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

