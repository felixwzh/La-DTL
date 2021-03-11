# coding=utf-8
import sys
import time
import logging
import argparse
import random
import numpy as np
import cPickle as pkl
import tensorflow as tf
import data
import model_mmd
from utils import time_format, evaluate,gene_conlleval_file,gene_word_2_id_dict
from temp import crf_evaluation,crf_cur_evaluation,get_f1_score
import os

def mkdir(path):
    path = path.strip()
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        print path + 'already exists'
        return False


plt_save_cnt=0
cur_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
log_path='../log/mmd/log'+cur_time
mkdir(log_path)



# highest fi-score for test
max_f1_test=0.

# training step num
num_training_step=0



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_path+'/log', filemode='w')

    parser = argparse.ArgumentParser()

    # data
    # source domain data
    parser.add_argument('-data_s', '--data_file_s', type=str)    # training and testing data file
    parser.add_argument('--n_shuffle_s', action='store_true')  # stands for not shuffle
    parser.add_argument('-bs_s', '--batch_size_s', type=int, default=128)
    # target domain data
    parser.add_argument('-data_t', '--data_file_t', type=str)  # training and testing data file
    parser.add_argument('--n_shuffle_t', action='store_true')  # stands for not shuffle
    parser.add_argument('-bs_t', '--batch_size_t', type=int, default=128)
    # unlabeled data
    parser.add_argument('-data_u', '--data_file_u', type=str)  # training and testing data file
    parser.add_argument('--n_shuffle_u', action='store_true')  # stands for not shuffle
    parser.add_argument('-bs_u', '--batch_size_u', type=int, default=128)

    # train
    parser.add_argument('-n', '--n_epochs', type=int, default=200)  
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-opt', '--optimizer', default='Adagrad')  
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.03)
    parser.add_argument('-dr', '--decay_rate', type=float, default=0.95)
    parser.add_argument('-ds', '--decay_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=3664401926)

    # model_mmd
    parser.add_argument('--n_embed_init', action='store_true')  # stand for not using embedding file
    parser.add_argument('-ls', '--label_size', type=int, default=7)   
    parser.add_argument('-vs', '--vocabulary_size', type=int, default=5017)   
    parser.add_argument('-es', '--embedding_size', type=int, default=100)
    parser.add_argument('-nh', '--n_rnn_hidden', type=int, default=100)
    parser.add_argument('-mmd', '--mmd_param', type=float, default=0.02)
    parser.add_argument('-A', '--A_dim', type=int, default=100)
    parser.add_argument('-rt', '--alpha_rate', type=float, default=0.2)


    # norm
    parser.add_argument('-l2', '--l2_coefficient', type=float, default=0.001)  # l2 norm
    parser.add_argument('-l2_p', '--l2_coefficient_p', type=float, default=0.1)  # l2 norm
    parser.add_argument('-l2_p_b', '--l2_coefficient_p_bias', type=float, default=0.1)  # l2 norm
    parser.add_argument('-kp', '--keep_prob', type=float, default=0.5)  # dropout rate
    parser.add_argument('-name', '--training_name', type=str, default='None')  

    # save args
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = random.randint(0, 4294967295)

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_path = '../log/mmd/log+' + args.training_name + '+' + cur_time
    mkdir(log_path)
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_path + '/log', filemode='w')

    logging.info(args)
    if not args.quiet:
        print args

    with open(log_path+'/hyperparams.pkl', 'wb') as fout:
        pkl.dump(args, fout)

    file_for_gene_dict = [
        '../data/two_domain/all_dict', #
    ]

    _, _, dict_share_1, dict_share_2 = gene_word_2_id_dict(*file_for_gene_dict)

    # load data

    dataset_s = data.Data_s(args)
    dataset_t = data.Data_t(args)
    dataset_u = data.Data_u(args)

    # train
    tic0 = time.time()
    model_mmd.set_hyperparams(args)
    with tf.Graph().as_default():
        # set random seed
        tf.set_random_seed(args.seed)

        # source domain data and label
        x_s_ = tf.placeholder(tf.int32, [None, None])
        y_s_ = tf.placeholder(tf.int32, [None, None])

        # target domain data and label
        x_t_ = tf.placeholder(tf.int32, [None, None])
        y_t_ = tf.placeholder(tf.int32, [None, None])

        # target domain unlabeled data
        x_u_ = tf.placeholder(tf.int32, [None, None])
        y_u_ = tf.placeholder(tf.int32, [None, None])

        
        mask_s_ = tf.placeholder(tf.bool, [args.label_size, args.batch_size_s, None])
        mask_t_ = tf.placeholder(tf.bool, [args.label_size, args.batch_size_t, None])
        mask_u_ = tf.placeholder(tf.bool, [args.label_size, args.batch_size_u, None])

        msl_ = tf.placeholder(tf.int32, [])

        # source domain sequence length and max length
        seq_len_s_ = tf.placeholder(tf.int32, [None])
        msl_s_ = tf.placeholder(tf.int32, [])
        # target domain sequence length and max length
        seq_len_t_ = tf.placeholder(tf.int32, [None])
        msl_t_ = tf.placeholder(tf.int32, [])
        # unlabeled data sequence length and max length
        seq_len_u_ = tf.placeholder(tf.int32, [None])
        msl_u_ = tf.placeholder(tf.int32, [])


        keep_prob_ = tf.placeholder(tf.float32, [])


        f1_score_ = tf.placeholder(tf.float32,[])


        # s_ is emission matrix, A_ is transition matrix
        # h_s_,h_t_,h_u_ are the hidden vectors after a linear projection above the LSTM layer
        h_s_= model_mmd.Represent_learner_pure(x_s_, seq_len_s_, msl_s_, keep_prob_, False)
        h_t_= model_mmd.Represent_learner_pure(x_t_, seq_len_t_, msl_t_, keep_prob_, True)
        h_u_= model_mmd.Represent_learner_pure(x_u_, seq_len_u_, msl_u_, keep_prob_, True)



        # concat h_s_ and h_t_ to h_
        h_ = tf.concat([h_s_,h_t_],0)
        # concat seq_len_s and seq_len_t to seq_len_
        seq_len_=tf.concat([seq_len_s_,seq_len_t_],0)
        # concat y_s_ and y_t_ to y_
        y_ = tf.concat([y_s_,y_t_],0)



        l2_ = args.l2_coefficient * model_mmd.l2_norm()


        
        with tf.variable_scope('fc_crf'):
            n_in = args.n_rnn_hidden * 2
            n_out = args.label_size
            limit = np.sqrt(6.0 / (n_in + n_out))  
            w_crf_s = tf.get_variable('w_crf_s', [n_in, n_out], initializer=tf.random_uniform_initializer(-limit, limit))
            b_crf_s = tf.get_variable('b_crf_s', [n_out], initializer=tf.constant_initializer(0.0))

        s_trns_s_ = tf.matmul(h_s_, w_crf_s)+b_crf_s
        s_trns_s_ = tf.reshape(s_trns_s_, [-1, msl_s_, args.label_size])


        with tf.variable_scope('fc_crf'):
            n_in = args.n_rnn_hidden * 2
            n_out = args.label_size
            limit = np.sqrt(6.0 / (n_in + n_out))  
            w_crf_t = tf.get_variable('w_crf_t', [n_in, n_out], initializer=tf.random_uniform_initializer(-limit, limit))
            b_crf_t = tf.get_variable('b_crf_t', [n_out], initializer=tf.constant_initializer(0.0))

        s_trns_t_ = tf.matmul(h_t_, w_crf_t)+b_crf_t
        s_trns_t_ = tf.reshape(s_trns_t_, [-1, msl_t_, args.label_size])


        
        norm_initializer = tf.random_normal_initializer(stddev=0.01)
        with tf.variable_scope('A'):
            A_trns_s_ = tf.get_variable('A_trns_s_', [args.label_size, args.label_size], initializer=norm_initializer)
            A_trns_t_ = tf.get_variable('A_trns_t_', [args.label_size, args.label_size], initializer=norm_initializer)

        l2_trns_p_loss_ = args.l2_coefficient_p * (
            tf.nn.l2_loss(A_trns_s_ - A_trns_t_) + tf.nn.l2_loss(w_crf_s - w_crf_t)) + \
                          args.l2_coefficient_p_bias * tf.nn.l2_loss(b_crf_s - b_crf_t)

        tf.summary.scalar('l2_trns_loss_ ', l2_trns_p_loss_ )

        
        c_trns_s_, A_trns_s_ = model_mmd.loss_trns_A(s_trns_s_, y_s_, seq_len_s_, A_trns_s_)
        tf.summary.scalar('c_trns_s_', c_trns_s_)

        c_trns_t_, A_trns_t_ = model_mmd.loss_trns_A(s_trns_t_, y_t_, seq_len_t_, A_trns_t_)
        tf.summary.scalar('c_trns_t_', c_trns_t_)

        c_ = args.alpha_rate*c_trns_s_ + (1-args.alpha_rate)*c_trns_t_

        

        mmd_loss_ = 0.

        _, _,_, _, _, _, _, mmd_loss_ = tf.while_loop(
            cond=lambda i, _1, _2, _3,_4, _5, label_size, _6: i < label_size,
            body=model_mmd.mmd_loss_label_aligned,
            loop_vars=(tf.constant(0, dtype=tf.int32),h_s_,h_t_,mask_s_,mask_t_,args.mmd_param,args.label_size,mmd_loss_),
            parallel_iterations=args.label_size
        )
        tf.summary.scalar('mmd_loss', mmd_loss_)




        loss_ = c_ + l2_ + mmd_loss_ + l2_trns_p_loss_

        # test/div

        s_test_ = tf.matmul(h_u_, w_crf_t)
        s_test_ = tf.reshape(s_test_, [-1, msl_u_, args.label_size])
        c_test_, A_trns_t_ = model_mmd.loss_trns_A(s_test_, y_u_, seq_len_u_, A_trns_t_)
        loss_test_= c_test_+l2_

        
        s_s_ = tf.matmul(h_s_, w_crf_s)
        s_s_ = tf.reshape(s_s_, [-1, msl_s_, args.label_size])
        c_s_, A_trns_s_ = model_mmd.loss_trns_A(s_s_, y_s_, seq_len_s_, A_trns_s_)
        loss_s_ = c_s_ + l2_

        s_t_ = tf.matmul(h_t_, w_crf_t)
        s_t_ = tf.reshape(s_t_, [-1, msl_t_, args.label_size])
        c_t_, A_trns_t_ = model_mmd.loss_trns_A(s_t_, y_t_, seq_len_t_, A_trns_t_)
        loss_t_ = c_t_ + l2_

        # train_op
        train_op, global_step = model_mmd.train(loss_)

        logging.info('\n' + '='*50)
        logging.info('Trainable Parameters')
        for param in tf.trainable_variables():
            logging.info(' ' + param.name)
        logging.info('='*50 + '\n')

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=20)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)


        sess.run(tf.global_variables_initializer())

        # tensor board
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path + '/', sess.graph)



        logging.info('Model builded, %s used\n' % time_format(time.time() - tic0))
        
        test_feed_dicts = []
        for i in xrange((len(dataset_u.test_x) + args.batch_size_u - 1) // args.batch_size_u):
            x, y, l, msl = data.pack(
                dataset_u.test_x[i*args.batch_size_u: (i+1)*args.batch_size_u],
                dataset_u.test_y[i*args.batch_size_u: (i+1)*args.batch_size_u]
            )
            test_feed_dicts.append({x_u_: x, seq_len_u_: l, y_u_: y, msl_u_: msl, keep_prob_: 1})




        
        train_s_feed_dicts = []
        for i in xrange((len(dataset_s.train_x) + args.batch_size_s - 1) // args.batch_size_s):
            x, y, l, msl = data.pack(
                dataset_s.train_x[i * args.batch_size_s: (i + 1) * args.batch_size_s],
                dataset_s.train_y[i * args.batch_size_s: (i + 1) * args.batch_size_s]
            )
            train_s_feed_dicts.append({x_s_: x, seq_len_s_: l, y_s_: y, msl_s_: msl, keep_prob_: 1})



        train_t_feed_dicts = []
        for i in xrange((len(dataset_t.train_x) + args.batch_size_t - 1) // args.batch_size_t):
            x, y, l, msl = data.pack(
                dataset_t.train_x[i * args.batch_size_t: (i + 1) * args.batch_size_t],
                dataset_t.train_y[i * args.batch_size_t: (i + 1) * args.batch_size_t]
            )
            train_t_feed_dicts.append({x_t_: x, seq_len_t_: l, y_t_: y, msl_t_: msl, keep_prob_: 1})


        valid_feed_dicts = []
        for i in xrange((len(dataset_t.test_x) + args.batch_size_u - 1) // args.batch_size_u):
            x, y, l, msl = data.pack(
                dataset_t.test_x[i * args.batch_size_u: (i + 1) * args.batch_size_u],
                dataset_t.test_y[i * args.batch_size_u: (i + 1) * args.batch_size_u]
            )
            valid_feed_dicts.append({x_t_: x, seq_len_t_: l, y_t_: y, msl_t_: msl, keep_prob_: 1})


        for epoch in xrange(1, args.n_epochs+1):
            tic = time.time()
            loss = 0.
            c_loss = 0.
            l2_loss = 0.
            mmd_loss = 0.
            n_train = dataset_t.n_train
            n_trained = 0


            pred_train_y_s = []

            pred_train_y_t = []




            for idx in dataset_t.minibatches:


                # find the largest msl in 3 domains
                msl_s = dataset_s.next_batch_msl()
                msl_t = dataset_t.next_batch_msl()
                msl_u = dataset_u.next_batch_msl()

                msl=max([msl_s,msl_t,msl_u])




                x_s, y_s, l_s, _,mask_s = dataset_s.next_batch_label_mask(msl,args)
                x_t, y_t, l_t, _,mask_t = dataset_t.next_batch_label_mask(msl,args)
                x_u, y_u, l_u, _,_ = dataset_u.next_batch_label_mask(msl,args)

                train_feed_dict = {
                    x_s_: x_s, seq_len_s_: l_s, y_s_: y_s,msl_s_:msl,
                    x_t_: x_t, seq_len_t_: l_t, y_t_: y_t,msl_t_:msl,
                    x_u_: x_u, seq_len_u_: l_u, y_u_: y_u,
                    msl_u_:msl,
                    msl_: msl, keep_prob_: args.keep_prob,
                    mask_s_:mask_s,
                    mask_t_:mask_t
                }


                if epoch!=1:
                    _, lossi, ci,l2i, mmdi, s_s, s_t = sess.run(
                        [train_op, loss_, c_,l2_, mmd_loss_, s_s_, s_t_],
                        feed_dict=train_feed_dict)
                else:
                    lossi, ci,l2i, mmdi, s_s, s_t = sess.run(
                        [loss_,  c_, l2_, mmd_loss_, s_s_, s_t_],
                        feed_dict=train_feed_dict)
                result = sess.run(merged, feed_dict=train_feed_dict)
                writer.add_summary(result, num_training_step)
                num_training_step+=1

                if np.isnan(lossi):
                    logging.error('Gradient Explosion!')
                    print 'Gradient Explosion!'
                    print 'ci',ci
                    print 'l2i',l2i
                    print 'loss-mmd',mmdi

                    exit()

                loss += lossi
                c_loss += ci
                l2_loss += l2i
                mmd_loss+=mmdi


                n_trained += len(idx)
                if not args.quiet:
                    print '[training] epoch %i >> %2.2f%% [%f=%f+%f+%f], completed in %s << \r' % (
                        epoch, n_trained * 100. / n_train, lossi, ci, l2i,mmdi,
                        time_format((time.time() - tic) * (n_train - n_trained) / n_trained)),
                    sys.stdout.flush()

            loss /= len(dataset_t.minibatches)
            c_loss /= len(dataset_t.minibatches)
            l2_loss /= len(dataset_t.minibatches)
            mmd_loss /= len(dataset_t.minibatches)

            logging.info('[training] epoch %i >> loss = %f , c_loss = %f, l2_loss = %f, mmd_loss = %f, %s [%s] used' %
                         (epoch, loss, c_loss, l2_loss, mmd_loss, time_format(time.time() - tic),
                          time_format(time.time() - tic0)))
            if not args.quiet:
                print '[training] epoch %i >> loss = %f , c_loss = %f, l2_loss = %f, mmd_loss = %f, %s [%s] used' % (
                    epoch, loss, c_loss, l2_loss, mmd_loss, time_format(time.time() - tic), time_format(time.time() - tic0))


            if epoch % 1 == 0:
                saver.save(sess, log_path + '/model', global_step=epoch)




            # calculate accuracy and loss in test data

            pred_y = []
            h_test_list = []
            y_test_list = []
            loss_t = 0.
            c_test_avrg = 0.
            for test_feed_dict in test_feed_dicts:
                s, A, c ,c_test,h_test= sess.run([s_test_, A_trns_t_, loss_test_,c_test_,h_u_], feed_dict=test_feed_dict)
                h_test_list.append(h_test)
                y_test_list.append(test_feed_dict[y_u_])
                loss_t += c
                c_test_avrg += c_test
                l = test_feed_dict[seq_len_u_]
                pred_y_i = model_mmd.predict(s, A, l)
                pred_y.extend(pred_y_i)  # pred_y is a two dimension array
            loss_t /= len(test_feed_dicts)
            c_test_avrg /= len(test_feed_dicts)
            l2_loss_cur=loss_t-c_test_avrg
            acc_test = evaluate(pred_y, dataset_u.test_y)

            if not args.quiet:
                rd = random.randrange(len(pred_y))
                print dataset_u.test_y[rd]
                print pred_y[rd]
                print '[valid] accuracy: %2.2f%%  loss: %f' % (acc_test, loss_t)



            # dump the whole train and test dataset_u with their prediction
            if epoch % 1 == 0:
                # f1 on test_data
                l_result = []
                l_result.append(dataset_u.test_x), l_result.append(dataset_u.test_y), l_result.append(pred_y)
                with open(log_path + '/test_result_%d.pkl' % epoch, 'wb') as fo:
                    pkl.dump(l_result, fo)

                gene_conlleval_file(log_path + '/test_result_%d.pkl' % epoch,
                                    log_path + '/eval_test_%d' % epoch, dict_share_1, dict_share_2)

                # eval
                crf_evaluation(epoch, epoch+1, 1, 'test', log_path + '/')
                crf_cur_evaluation(epoch, 'test', log_path + '/')

                #get f1-score
                f1_score=get_f1_score(epoch, 'test', log_path + '/')
                if f1_score>max_f1_test:
                    max_f1_test=f1_score
                print '[valid] f1-score-cur: %f' % f1_score
                print '[valid] f1-score-max: %f' % max_f1_test
                logging.info('[valid] accuracy: %2.2f%%  c_loss: %f f1-score: %f l2_loss: %f' % (
                acc_test, c_test_avrg, f1_score, l2_loss_cur))





                pred_y_valid = []
                h_valid_list = []
                y_valid_list = []
                loss_t = 0.
                for valid_feed_dict in valid_feed_dicts:
                    s, A, c, h_t = sess.run([s_t_, A_trns_t_, c_t_, h_t_], feed_dict=valid_feed_dict)

                    h_valid_list.append(h_t)
                    y_valid_list.append(valid_feed_dict[y_t_])

                    loss_t += c
                    l = valid_feed_dict[seq_len_t_]
                    pred_y_i = model_mmd.predict(s, A, l)
                    pred_y_valid.extend(pred_y_i)  
                loss_t /= len(valid_feed_dicts)
                acc_test = evaluate(pred_y_valid, dataset_t.test_y)

                l_result = []
                l_result.append(dataset_t.test_x), l_result.append(dataset_t.test_y), l_result.append(pred_y_valid)
                with open(log_path + '/valid_result_%d.pkl' % epoch, 'wb') as fo:
                    pkl.dump(l_result, fo)

                gene_conlleval_file(log_path + '/valid_result_%d.pkl' % epoch,
                                    log_path + '/eval_valid_%d' % epoch, dict_share_1, dict_share_2)

                crf_evaluation(epoch, epoch + 1, 1, 'valid', log_path + '/')
                crf_cur_evaluation(epoch, 'valid', log_path + '/')

                f1_score = get_f1_score(epoch, 'valid', log_path + '/')
                logging.info('[test] accuracy: %2.2f%%  c_loss: %f f1-score: %f' % (acc_test, loss_t, f1_score))
