# coding=utf-8
import subprocess

def crf_evaluation(idx_start, idx_end, step, file_type, PATH):
    for epoch in range(idx_start, idx_end, step):
        filename = PATH + 'eval_%s_%d' % (file_type, epoch)
        # print filename
        eval_test = to_conll(filename)
        p_r_f_v, _ = subprocess.Popen(['perl ../eval/conlleval.pl -d \\\t'], shell=True, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE).communicate(eval_test.encode('utf8'))
        with open(PATH + '000_eval_%s_result' % file_type, 'a') as f:
            f.write(p_r_f_v.decode())

def crf_cur_evaluation(idx, file_type, PATH):
    epoch = idx
    filename = PATH + 'eval_%s_%d' % (file_type, epoch)
    # print filename
    eval_test = to_conll(filename)
    p_r_f_v, _ = subprocess.Popen(['perl ../eval/conlleval.pl -d \\\t'], shell=True, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE).communicate(eval_test.encode('utf8'))
    with open(PATH + 'cur_eval_%s_result' % file_type, 'w') as f:
        f.write(p_r_f_v.decode())

def get_f1_score(idx,file_type,PATH):
    epoch = []
    f1 = []
    highest_f1 = []
    highest_idx = []
    # for file_type in ['train', 'test']:
    with open(PATH+'cur_eval_%s_result' % file_type) as fin:
        max_f1 = 0.0
        max_idx = 0
        for line in fin:
            s = line.strip().split()
            if s[0] == 'accuracy:':
                epoch.append(idx)
                curr = float(s[7])
                f1.append(curr)
                if curr > max_f1:
                    max_f1 = curr
                    max_idx = idx

    highest_f1.append(max_f1)
    highest_idx.append(max_idx)

    return f1[0]



