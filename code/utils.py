# coding=utf-8
import numpy as np
import cPickle as pkl
def time_format(t):
    s = ''
    if t >= 3600:
        s += '%d h ' % (t / 3600)
        t %= 3600
    if t >= 60:
        s += '%d m ' % (t / 60)
        t %= 60
    s += '%d s' % t
    return s

def evaluate(pred_y, y):
    correct_labels = 0
    total_labels = 0
    for y1, y2 in zip(pred_y, y):
        y2 = y2[:len(y1)]
        assert len(y1) == len(y2)
        correct_labels += np.sum(np.equal(y1, y2))
        total_labels += len(y1)
    accuracy = 100.0 * correct_labels / float(total_labels)

    return accuracy

def gene_conlleval_file(input_file, output_file, dict_1, dict_2):
    x, y, p = pkl.load(open(input_file, 'rb'))
    d1, d2 = dict_1, dict_2
    # print 'write to file: ' + output_file
    with open(output_file, 'w') as fout:
        for xi, yi, pi in zip(x, y, p):
            xi = xi[:len(pi)]
            yi = yi[:len(pi)]
            # assert len(xi) == len(yi)
            # assert len(xi) == len(pi)
            for j in range(len(xi)):
                fout.write(d1[xi[j]] + '\t' + d2[yi[j]] + '\t' + d2[pi[j]] + '\n')
            fout.write('\n')
# gene word-id dict from ctf++ format file
def gene_word_2_id_dict(*filenames):
    word_2_id_dict = {}
    label_2_id_dict = {}
    id_2_word_dict = {}
    id_2_label_dict = {}
    word_idx = 0
    label_idx = 0
    for filename in filenames:
        print 'dict_file', filename
        with open(filename, 'r') as fi:
            for line in fi:
                s = line.strip().split()
                assert len(s) == 0 or len(s) == 2
                if len(s) == 2:
                    if not s[0] in word_2_id_dict.keys():
                        word_2_id_dict[s[0]] = word_idx
                        id_2_word_dict[word_idx] = s[0]
                        word_idx += 1
                    if not s[1] in label_2_id_dict.keys():
                        label_2_id_dict[s[1]] = label_idx
                        id_2_label_dict[label_idx] = s[1]
                        label_idx += 1
                else:
                    pass
    return word_2_id_dict, label_2_id_dict, id_2_word_dict, id_2_label_dict
