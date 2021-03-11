#encoding=utf-8
import cPickle as pkl
import numpy as np

class Data:

    def __init__(self, args):
        with open(args.data_file, 'rb') as fin:  # interface between whole model and training data
            self.train_x, self.train_y, self.test_x, self.test_y = pkl.load(fin)
        self.shuffle = not args.n_shuffle
        self.batch_size = args.batch_size

        np.random.seed(args.seed)  # set the random seed for numpy model using args.seed

        self.n_train = len(self.train_x)

        self.generate_minibatches()

    def generate_minibatches(self):
        idx_list = np.arange(self.n_train, dtype='int32') # generate a idx_list for training
        if self.shuffle:
            np.random.shuffle(idx_list)

        # re-generate the whole minibatches
        self.minibatches = []
        self.minibatch_idx = 0

        idx = 0
        for i in xrange(self.n_train // self.batch_size):
            self.minibatches.append(idx_list[idx:idx+self.batch_size])
            idx += self.batch_size
        if idx != self.n_train:
            self.minibatches.append(idx_list[idx:])

    def next_batch(self):
        # after a epoch, re-generate the minibatches
        if self.minibatch_idx >= len(self.minibatches):
            self.generate_minibatches()

        # put the data by the idx
        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        y = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
        x, y, l, msl = pack(x, y)
        self.minibatch_idx += 1
        return x, y, l, msl

class Data_s:

    def __init__(self, args):
        with open(args.data_file_s, 'rb') as fin:  # interface between whole model and training data
            self.train_x, self.train_y, self.test_x, self.test_y = pkl.load(fin)
        self.shuffle = not args.n_shuffle_s
        self.batch_size = args.batch_size_s

        np.random.seed(args.seed)  # set the random seed for numpy model using args.seed

        self.n_train = len(self.train_x)

        self.generate_minibatches()





    def generate_minibatches(self):
        idx_list = np.arange(self.n_train, dtype='int32') # generate a idx_list for training
        if self.shuffle:
            np.random.shuffle(idx_list)

        # re-generate the whole minibatches
        self.minibatches = []
        self.minibatch_idx = 0

        idx = 0
        for i in xrange(self.n_train // self.batch_size):
            self.minibatches.append(idx_list[idx:idx+self.batch_size])
            idx += self.batch_size

    def next_batch_msl(self):
        if self.minibatch_idx >= len(self.minibatches):
            self.generate_minibatches()

        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        l = map(len, x)  # true length of each training sentence
        msl = max(l)
        return msl

    def next_batch(self,msl):


        # put the data by the idx
        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        y = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
        x, y, l, msl = pack_mmd_0(x, y, msl)
        self.minibatch_idx += 1
        return x, y, l, msl


    def next_batch_label_mask(self,msl,args):

        # put the data by the idx
        x_0 = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        y_0 = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
        x, y, l, msl = pack_mmd(x_0, y_0, msl)
        mask=gen_label_mask(y=y, label_size=args.label_size, batch_size=args.batch_size_s, msl=msl)
        self.minibatch_idx += 1
        x, y, _, _ = pack_mmd_0(x_0, y_0, msl)
        return x, y, l, msl,mask


class Data_t:
    def __init__(self, args):
        with open(args.data_file_t, 'rb') as fin:  # interface between whole model and training data
            self.train_x, self.train_y, self.test_x, self.test_y = pkl.load(fin)
        self.shuffle = not args.n_shuffle_t
        self.batch_size = args.batch_size_t

        np.random.seed(args.seed)  # set the random seed for numpy model using args.seed

        self.n_train = len(self.train_x)

        self.generate_minibatches()

    def generate_minibatches(self):
        idx_list = np.arange(self.n_train, dtype='int32')  # generate a idx_list for training
        if self.shuffle:
            np.random.shuffle(idx_list)

        # re-generate the whole minibatches
        self.minibatches = []
        self.minibatch_idx = 0

        idx = 0
        for i in xrange(self.n_train // self.batch_size):
            self.minibatches.append(idx_list[idx:idx + self.batch_size])
            idx += self.batch_size


    def next_batch_msl(self):
        # after a epoch, re-generate the minibatches
        if self.minibatch_idx >= len(self.minibatches):
            self.generate_minibatches()
        # print self.minibatches
        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        l = map(len, x)  # true length of each training sentence
        msl = max(l)
        return msl

    def next_batch(self, msl):

        # put the data by the idx
        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        y = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
        x, y, l, msl = pack_mmd_0(x, y, msl)
        self.minibatch_idx += 1
        return x, y, l, msl

    def next_batch_label_mask(self,msl,args):

        # put the data by the idx
        x_0 = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        y_0 = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
        x, y, l, msl = pack_mmd(x_0, y_0, msl)
        mask=gen_label_mask(y=y, label_size=args.label_size, batch_size=args.batch_size_t, msl=msl)
        self.minibatch_idx += 1
        x, y, _, _ = pack_mmd_0(x_0, y_0, msl)
        return x, y, l, msl,mask


class Data_u:
    def __init__(self, args):
        with open(args.data_file_u, 'rb') as fin:  # interface between whole model and training data
            self.train_x, self.train_y, self.test_x, self.test_y = pkl.load(fin)
        self.shuffle = not args.n_shuffle_u
        self.batch_size = args.batch_size_u

        np.random.seed(args.seed)  # set the random seed for numpy model using args.seed

        self.n_train = len(self.train_x)

        self.generate_minibatches()

    def generate_minibatches(self):
        idx_list = np.arange(self.n_train, dtype='int32')  # generate a idx_list for training
        if self.shuffle:
            np.random.shuffle(idx_list)

        # re-generate the whole minibatches
        self.minibatches = []
        self.minibatch_idx = 0

        idx = 0
        for i in xrange(self.n_train // self.batch_size):
            self.minibatches.append(idx_list[idx:idx + self.batch_size])
            idx += self.batch_size

    def next_batch_msl(self):
        # after a epoch, re-generate the minibatches
        if self.minibatch_idx >= len(self.minibatches):
            self.generate_minibatches()

        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        l = map(len, x)  # true length of each training sentence
        msl = max(l)
        return msl

    def next_batch(self, msl):

        # put the data by the idx
        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        y = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
        x, y, l, msl = pack_mmd_0(x, y, msl)
        self.minibatch_idx += 1
        return x, y, l, msl

    def next_batch_label_mask(self,msl,args):

        # put the data by the idx
        x_0 = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        y_0 = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
        x, y, l, msl = pack_mmd(x_0, y_0, msl)
        mask=gen_label_mask(y=y, label_size=args.label_size, batch_size=args.batch_size_u, msl=msl)
        self.minibatch_idx += 1
        x, y, _, _ = pack_mmd_0(x_0, y_0, msl)
        return x, y, l, msl,mask

class Data_test:


    def __init__(self, data_file_s,n_shuffle_s,batch_size_s,seed):
        with open(data_file_s, 'rb') as fin:  # interface between whole model and training data
            self.train_x, self.train_y, self.test_x, self.test_y = pkl.load(fin)
        self.shuffle = not n_shuffle_s
        self.batch_size = batch_size_s

        np.random.seed(seed)  # set the random seed for numpy model using args.seed

        self.n_train = len(self.train_x)

        self.generate_minibatches()

    def generate_minibatches(self):
        idx_list = np.arange(self.n_train, dtype='int32') # generate a idx_list for training
        if self.shuffle:
            np.random.shuffle(idx_list)

        # re-generate the whole minibatches
        self.minibatches = []
        self.minibatch_idx = 0

        idx = 0
        for i in xrange(self.n_train // self.batch_size):
            self.minibatches.append(idx_list[idx:idx+self.batch_size])
            idx += self.batch_size
        if idx != self.n_train:
            self.minibatches.append(idx_list[idx:])

    def next_batch_msl(self):
        # after a epoch, re-generate the minibatches
        if self.minibatch_idx >= len(self.minibatches):
            self.generate_minibatches()

        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        l = map(len, x)  # true length of each training sentence
        msl = max(l)
        return msl

    def next_batch(self,msl):


        # put the data by the idx
        x = [self.train_x[i] for i in self.minibatches[self.minibatch_idx]]
        y = [self.train_y[i] for i in self.minibatches[self.minibatch_idx]]
        x, y, l, msl = pack_mmd_0(x, y, msl)
        self.minibatch_idx += 1
        return x, y, l, msl

def pack_mmd(x_raw, y_raw, msl):
    l = map(len, x_raw)  # true length of each training sentence
    n = len(l)
    x = np.zeros([n, msl], dtype='int32')
    y = np.full([n, msl], -1, dtype='int32')

    for i, a in enumerate(x_raw):
        x[i, :len(a)] = np.asarray(a)
    for i, a in enumerate(y_raw):
        y[i, :len(a)] = np.asarray(a)

    return x, y, l, msl

def pack_mmd_0(x_raw, y_raw, msl):
    l = map(len, x_raw)  # true length of each training sentence
    n = len(l)
    x = np.zeros([n, msl], dtype='int32')
    y = np.full([n, msl], 0, dtype='int32')

    for i, a in enumerate(x_raw):
        x[i, :len(a)] = np.asarray(a)
    for i, a in enumerate(y_raw):
        y[i, :len(a)] = np.asarray(a)

    return x, y, l, msl

def gen_label_mask(y,label_size,batch_size,msl):
    """
    :param y: a [batch_size * max_seq_len] array, padded with -1
    :param label_size:  number of different labels
    :return: a label mask of shape [ label_size * batch_size * max_seq_len ]
    """
    def f(x, y, z):
        return x
    mask_origin = np.fromfunction(f, (label_size,batch_size,msl))

    mask_y=np.array([y]*label_size)





    if mask_origin.shape !=mask_y.shape:
      print mask_origin.shape
      print mask_y.shape
      raise ValueError("mask_origin and mask_y should have the same shape! ")

    mask=np.equal(mask_origin,mask_y)

    return mask


def pack(x_raw, y_raw):
    l = map(len, x_raw)  # true length of each training sentence
    msl = max(l)
    n = len(l)
    x = np.zeros([n, msl], dtype='int32')
    y = np.full([n, msl], 0, dtype='int32')
    for i, a in enumerate(x_raw):
        x[i, :len(a)] = np.asarray(a)
    for i, a in enumerate(y_raw):
        y[i, :len(a)] = np.asarray(a)

    return x, y, l, msl

