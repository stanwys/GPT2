import encoder
import tensorflow as tf
from model import GPT2Model
from utils import default_hparams, top_k_logits, past_shape
from load_dataset import my_load_dataset, Sampler
import sys
import time

MIN_TOKENS_IN_SEQUENCE = 100

def getBatch(data_sampler,batch_size,context_size):
    return [data_sampler.sample(context_size) for _ in range(batch_size)]

def train(
        batch_size = 2,
        num_epochs = 2,
        learning_rate = 0.001,
        num_tokens_sequence = 1024,
        save_model_every_num_iter = 100,
        input_path = "data/small-117Mtest.txt",
        read_initial_weights=False,
        save_weights = True,
        n_layers = 12,
        weights_path_read = "./weights/checkpoint",
        weights_path_write = "./weights/checkpoint",
        show_every = 10):

    hparams = default_hparams()
    hparams.n_layer = n_layers
    model = GPT2Model(hparams)
    num_tokens = max(min(hparams.context_length[0],num_tokens_sequence),MIN_TOKENS_IN_SEQUENCE)

    if read_initial_weights == True:
        model.load_weights(weights_path_read)

    enc = encoder.get_encoder("model117")
    chunks = my_load_dataset(enc, input_path, 50000)
    data_sampler = Sampler(chunks)

    num_iterations_in_epoch = data_sampler.total_size // (num_tokens * batch_size)

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    avg_loss = (0.0, 0.0)
    start_time = time.time()
    print("Total epochs: ",num_epochs)
    print("Num of iterations per epoch: ", num_iterations_in_epoch)
    print("Batch size: ", batch_size)
    print("Sequence length(in tokens): ", num_tokens)

    counter = 0
    for epoch in range(num_epochs):
        print("Epoch: ",epoch)
        for iteration in range(num_iterations_in_epoch):
            batch = getBatch(data_sampler, batch_size, num_tokens)
            context_tensor = tf.reshape(tf.convert_to_tensor(batch, dtype=tf.int64), (batch_size, num_tokens))
            with tf.GradientTape() as tape:

                out = model(context_tensor, past=None)
                loss = loss_function(context_tensor, out['logits'])

                '''
                #batch in form of for loop to solve problems with lack of memory
                tensors = []
                #for tokens in batch:
                for _ in range(batch_size):
                    tokens = getBatch(data_sampler, 1, num_tokens)
                    context_tensor = tf.reshape(tf.convert_to_tensor(tokens, dtype=tf.int64), (1, num_tokens))
                    out = model(context_tensor, past=None)
                    loss_tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=context_tensor,logits=out["logits"])
                    tensors.append(loss_tmp)

                loss = tf.reduce_mean(tf.stack(tensors, axis = 1))
                '''
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            avg_loss = (avg_loss[0] * 0.99 + loss,
                        avg_loss[1] * 0.99 + 1.0)
            if(counter % show_every == 0 or iteration == num_iterations_in_epoch - 1):
                print(
                    '[{counter} of {num_iters} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                        .format(
                        counter=iteration,
                        num_iters = num_iterations_in_epoch,
                        time=time.time() - start_time,
                        loss=loss,
                        avg=avg_loss[0] / avg_loss[1]))
            counter += 1
            if((counter) % save_model_every_num_iter == 0 and save_weights == True):
                model.save_weights(weights_path_write)

    model.save_weights(weights_path_write)

if __name__ == "__main__":
    p_batch_size = int(sys.argv[1])
    p_num_epochs = int(sys.argv[2])
    p_learning_rate = float(sys.argv[3])
    p_num_tokens_sequence = int(sys.argv[4])
    p_read_initial_weights = bool(int(sys.argv[5]))
    p_save_weights = bool(int(sys.argv[6]))
    p_save_model_every_num_iter = int(sys.argv[7])
    p_n_layers = int(sys.argv[8])
    p_weights_path_read = sys.argv[9]#"./weights/checkpoint",
    p_weights_path_write = sys.argv[10]#"./weights/checkpoint",
    p_input_path = sys.argv[11]#"data/small-117Mtest.txt",
    p_show_every = int(sys.argv[12])
    train(batch_size=p_batch_size,
          num_epochs=p_num_epochs,
          learning_rate=p_learning_rate,
          num_tokens_sequence=p_num_tokens_sequence,
          read_initial_weights=p_read_initial_weights,
          save_weights=p_save_weights,
          save_model_every_num_iter=p_save_model_every_num_iter,
          n_layers=p_n_layers,
          weights_path_read=p_weights_path_read,
          weights_path_write=p_weights_path_write,
          input_path=p_input_path,
          show_every=p_show_every)#batch_size= batch_size, save_weights = save_weights)