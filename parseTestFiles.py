import numpy as np
import encoder

def start_pos(n):
    if n < 10:
        return 2
    return 3

def getContextTokens(encoder, raw_context, max_context_length = 1023):
    raw_tokens = encoder.encode(raw_context)
    tokens_length = min(len(raw_tokens), max_context_length)
    return np.stack(raw_tokens[-tokens_length:])

def getAllAnswers(final_line):
    strings1 = final_line.replace('\n',"").split(" ")
    possible_answers_pos = -1
    while('|' not in strings1[possible_answers_pos]):
        possible_answers_pos -= 1
    strings2 = strings1[possible_answers_pos].split('\t')
    possible_answers = strings2[-1].split('|')
    answer = strings2[-3]

    return answer, possible_answers

def getAllAnswerTokens(encoder, answer, possible_answers):
    answer_tokens = encoder.encode(answer)
    possible_answers_tokens = []
    for pos_answer in possible_answers:
        possible_answers_tokens.append(encoder.encode(pos_answer))

    return np.stack(answer_tokens), possible_answers_tokens

def parse_test_file(mode, input_path,max_context_length = 1023):
    returned_data = {
        'contexts' : [],
        'answers' : [],
        'possible_answers' : [],
        'num_examples' : 0
    }
    enc = encoder.get_encoder("model117")

    f = open(input_path)
    text_lines = f.readlines()
    f.close()
    if (mode == "CBT"):
        counter = 1
        raw_context = ''
        for line in text_lines:
            if(counter <= 21):
                raw_line = line[ start_pos(counter):]
                if(counter < 21):
                    raw_context += raw_line
                elif(counter == 21):
                    XXXXX_pos = raw_line.find("XXXXX")
                    raw_context += raw_line[:XXXXX_pos]
                    answer, possible_answers = getAllAnswers(raw_line)
                    context_tokens = getContextTokens(enc, raw_context, max_context_length)
                    answer_tokens, possible_answers_tokens = getAllAnswerTokens(enc, answer, possible_answers)
                    returned_data['contexts'].append(context_tokens)
                    returned_data['answers'].append(answer_tokens)
                    returned_data['possible_answers'].append(possible_answers_tokens)
                    returned_data['num_examples'] += 1

                counter += 1
            else:
                counter = 1
                raw_context = ''

    elif (mode == "LAMBADA"):
        for line in text_lines:
            answer = line.split(" ")[-1]
            answer_tokens = enc.encode(answer)
            answer_last_pos = line.rfind(answer)
            context_tokens = getContextTokens(enc, line[:answer_last_pos], max_context_length)
            returned_data['contexts'].append(context_tokens)
            returned_data['answers'].append(answer_tokens)
            returned_data['num_examples'] += 1

    return returned_data
