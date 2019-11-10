# 导入数据
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
with open('data/data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    id2word = dict(id2word)
# 划分测试集/训练集/验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  #原0.2
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)   #原0.2
print('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
    X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))


# ** 3.build the data generator
class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.

    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """

    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]


print('Creating the data generator ...')
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
print('Finished creating the data generator.')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np
import time
'''
For Chinese word segmentation.
'''
# ##################### config ######################
decay = 0.85
max_epoch = 7
max_max_epoch = 10
timestep_size = max_len = 100  # 句子长度
vocab_size = 8912 # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64  # 字向量长度
class_num = 3
hidden_size = 128  # 隐含层节点数
layer_num = 2  # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
model_save_path = 'ckpt/bi-lstm.ckpt'  # 模型保存位置

with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def bi_lstm(X_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    inputs = tf.nn.embedding_lookup(embedding, X_inputs)

    # ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)

    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)


    # ** 3. bi-lstm 计算（展开）
    with tf.variable_scope('bidirectional_rnn'):
        # *** 下面，两个网络是分别计算 output 和 state
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)

        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        # *** 然后把 output_bw 在 timestep 维度进行翻转
        # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
        outputs_bw = tf.reverse(outputs_bw, [0])
        # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.reshape(output, [-1, hidden_size * 2])
    # ***********************************************************
    return output  # [-1, hidden_size*2]


with tf.variable_scope('Inputs'):
    X_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='X_input')
    y_inputs = tf.placeholder(tf.int32, [None, timestep_size], name='y_input')
    wordNum = tf.placeholder(tf.int32,name='wordNum')
    allword = tf.placeholder(tf.int32,name='allword')
bilstm_output = bi_lstm(X_inputs)

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_size * 2, class_num])
    softmax_b = bias_variable([class_num])
    y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b
# adding extra statistics to monitor
# y_inputs.shape = [batch_size, timestep_size]

correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
ypre = tf.cast(tf.argmax(y_pred, 1), tf.int32)
accuracy=(tf.reduce_sum(tf.cast(correct_prediction,tf.int32))+wordNum-allword)/wordNum
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))        ##############################################

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]), logits=y_pred))

# ***** 优化求解 *******
tvars = tf.trainable_variables()  # 获取模型的所有参数
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # 优化器

# 梯度下降计算
train_op = optimizer.apply_gradients(zip(grads, tvars),
                                     global_step=tf.train.get_or_create_global_step())
print('Finished creating the bi-lstm model.')


#########################训练模型######################
def test_epoch(dataset):
    """Testing or valid."""
    _batch_size = 500
    fetches = [accuracy, cost, y_pred]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    for i in range(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)

        # print(X_batch[0])                #########################查看x，y 后面的 0 。
        # print("hhhh:",np.shape(y_batch))                #(500*32)每批500句话，每句话32个字。
        wordnum = len(X_batch.nonzero()[0])
        aword = _batch_size * timestep_size
        # print(wordnum)        #一个批中500*32个字中，非零的个数。
        feed_dict = {X_inputs: X_batch, y_inputs: y_batch, lr: 1e-5, batch_size: _batch_size,allword:aword,wordNum: wordnum ,keep_prob:1.0}
        _acc, _cost,_predY = sess.run(fetches, feed_dict)  #送入的是feed_dict,输出fetches。
        # print(np.argmax(_predY[:32],1))
        # print(y_batch[0])
        # print(len(y_batch[0].nonzero()[0]))

        # print(_acc)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++")
        _accs += _acc                     ############################################################
        _costs += _cost
    mean_acc= _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost


sess.run(tf.global_variables_initializer())
tr_batch_size = 500
display_num = 10  # 每个 epoch 显示是个结果
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
f = open("log.log","w",encoding="utf-8")
for epoch in range(max_max_epoch):
    _lr = 1e-3
    if epoch > max_epoch:
        _lr = _lr * ((decay) ** (epoch - max_epoch))
    print('EPOCH %d， lr=%g' % (epoch+1, _lr))
    f.write("EPOCH"+str(epoch+1)+"\n")
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    for batch in range(tr_batch_num):
        fetches = [accuracy, cost, train_op, ypre]
        X_batch, y_batch = data_train.next_batch(tr_batch_size)
        wordnum = len(X_batch.nonzero()[0])
        aword = tr_batch_size*timestep_size
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, lr:_lr, batch_size:tr_batch_size,allword:aword,wordNum:wordnum, keep_prob:0.5}   #4096是batch100大小 乘以 每句话字的个数100。
        
        _acc, _cost, _ ,yp = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch#############################################
        
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        if (batch + 1) % display_batch == 0:
            xin = []
            for i in X_batch[0]:
                if i != 0:
                    xin.append(id2word[i])
            xinput = "".join(xin)

            print("xinput:",xinput)
            try:
                f.write(str(xinput)+"\n")
            except:
                pass
            print("yright:",y_batch[0])  ##########################################
            f.write(str(y_batch[0])+"\n")
            print("ypredt:",yp[:100])  ###################################
            f.write(str(yp[:100])+"\n")
            print("---------------------------\n")

            valid_acc, valid_cost = test_epoch(data_valid)  # valid
            print('\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                show_costs / display_batch, valid_acc, valid_cost))
            f.write("acc:"+str(show_accs / display_batch)+"====")
            f.write("cost:"+str(show_costs / display_batch)+"\n")
            show_accs = 0.0
            show_costs = 0.0
    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num
    if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print('the save path is ', save_path)
    print('\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
    print('Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time))
# testing
print('**TEST RESULT:')
test_acc, test_cost = test_epoch(data_test)
print('**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost))

saver = tf.train.Saver()
best_model_path = 'ckpt/bi-lstm.ckpt-6'
saver.restore(sess, best_model_path)
X_tt, y_tt = data_train.next_batch(1)
print('X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape)
print('X_tt = ', X_tt)
print('y_tt = ', y_tt)