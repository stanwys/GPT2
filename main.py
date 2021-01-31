import encoder
import tensorflow as tf
from model import GPT2Model
from utils import default_hparams, top_k_logits, past_shape
import sys

def generate_words(temperature, top_k, output_sequence_length, save_weights):

    hparams = default_hparams()
    model = GPT2Model(hparams)

    if(save_weights == 1):
        model.save_weights('./weights/checkpoint')
    else:
        model.load_weights('./weights/checkpoint')

    batch_size = 1

    enc = encoder.get_encoder("model117")
    raw_text = input("Podaj tekst: ")

    past = None
    context_tokens = enc.encode(raw_text)
    start_sequence_size = len(context_tokens)
    context_tensor = tf.reshape(tf.convert_to_tensor(context_tokens, dtype=tf.int64), (1, start_sequence_size))

    if start_sequence_size > 1:
        previous = context_tensor[:,:-1]#tf.reshape(context_tensor[:,:-1], (1, start_sequence_size-1))
        # generate first to get hidden past states at the beginning
        out = model.call(previous, past=past)
        past = out['present']
        previous = tf.reshape(context_tensor[:, -1], (1, 1))
    else:
        previous = context_tensor#tf.reshape(context_tensor, (1, 1))

    out_tensor = context_tensor

    final_sequence_length = max(min(output_sequence_length, hparams.context_length[0]), start_sequence_size + 1)

    for i in range(final_sequence_length - start_sequence_size):
        out = model.call(previous, past=past)
        logits = out['logits'][:, -1, :] / tf.cast(temperature,tf.float32)
        logits = top_k_logits(logits, k=top_k)
        samples = tf.random.categorical(logits, num_samples=1)

        presents = out['present']
        presents.set_shape(past_shape(hparams=hparams, batch_size=batch_size))

        if past is not None:
            past = tf.concat([past, presents], axis=-2)
        else:
            past = presents

        previous = tf.squeeze(samples, axis=[1])
        previous = tf.reshape(previous, (1, 1))
        out_tensor = tf.concat([out_tensor , samples], axis=1)

        raw_text = enc.decode(out_tensor.numpy()[0])
        print(raw_text)


if __name__ == "__main__":
    temperature = int(sys.argv[1])
    top_k = int(sys.argv[2])
    output_sequence_length = int(sys.argv[3])
    save_weights = int(sys.argv[4])
    generate_words(temperature, top_k, output_sequence_length,save_weights)



