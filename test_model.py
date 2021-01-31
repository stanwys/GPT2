import tensorflow as tf
from model import GPT2Model
from utils import default_hparams, top_k_logits
from parseTestFiles import parse_test_file
import sys
import numpy as np

def test_model(dataset_input_path,
               weights_input_path, results_output_path,
               results_output_mode, mode = "LAMBADA",
               max_context_length = 1023):
    test_data = parse_test_file(mode, dataset_input_path, max_context_length)

    hparams = default_hparams()
    model = GPT2Model(hparams)
    model.load_weights(weights_input_path)
    #temperature = 1
    top_k = 1
    correct_answers = 0
    given_answers = {}

    for index in range(test_data['num_examples']):
        if (index % 100 == 0):
            print("iteration: ",index," of ",test_data['num_examples'])

        context_tensor = tf.reshape(
            tf.convert_to_tensor(test_data['contexts'][index],dtype=tf.int64),
            (1, test_data['contexts'][index].shape[0]))

        previous = context_tensor[:, :-1]
        out = model(previous, past=None)
        past = out['present']
        previous = tf.reshape(context_tensor[:, -1], (1, 1))
        out = model(previous, past=past)

        if (mode == "LAMBADA"):
            logits = out['logits'][:, -1, :]# / tf.cast(temperature, tf.float32)
            logits = top_k_logits(logits, k=top_k)
            sample = tf.random.categorical(logits, num_samples=1)
            sample_token = sample.numpy()[0][0]
            if sample_token in given_answers.keys():
                given_answers[sample_token] += 1
            else:
                given_answers[sample_token] = 1
            if (sample_token in test_data['answers'][index]):
                correct_answers += 1

        elif (mode == "CBT"):
            best_sum = -np.inf
            best_tokens = np.array([])
            for possible_answers in test_data['possible_answers'][index]:
                logits_sum = 0
                for token in possible_answers:
                    logits_sum += out['logits'][:, -1, token].numpy()[0]
                if logits_sum > best_sum:
                    best_sum = logits_sum
                    best_tokens = np.array(possible_answers)

            true_tokens = test_data['answers'][index]
            if ( np.array_equal(true_tokens , best_tokens)):
                correct_answers += 1

    f = open(results_output_path, mode=results_output_mode)
    if results_output_mode == 'w':
        f.write("temperature;top_k;acc\n")
    f.write("{temp};{k};{acc:2.2f}".format(temp=1, k=1, acc=correct_answers/test_data['num_examples']))
    f.close()


if __name__ == "__main__":
    p_mode = sys.argv[1]
    p_dataset_input_path = sys.argv[2]
    p_weights_input_path = sys.argv[3]
    p_results_output_path = sys.argv[4]
    p_results_output_mode = sys.argv[5]
    p_max_context_length = int(sys.argv[6])
    test_model(mode = p_mode,
               dataset_input_path = p_dataset_input_path,
               weights_input_path = p_weights_input_path,
               results_output_path = p_results_output_path,
               results_output_mode = p_results_output_mode,
               max_context_length=p_max_context_length)