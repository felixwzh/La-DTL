CUDA_VISIBLE_DEVICES="0" python2 train.py -data_s ../data/new-weibo/sighan/eval_1_train.pkl -bs_s 15 -data_t ../data/new-weibo/weibo/eval_1_train.pkl -bs_t 15 -data_u ../data/new-weibo/weibo/eval_test.pkl  -bs_u 30