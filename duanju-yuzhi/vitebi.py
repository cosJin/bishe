# 导入数据
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import pickle
import os


#   https://github.com/yongyehuang/Tensorflow-Tutorial/blob/master/Tutorial_6%20-%20Bi-directional%20LSTM%20for%20sequence%20labeling%20(Chinese%20segmentation).ipynb  #
with open('cleaned_train800-1160.txt', 'rb') as inp:
    texts = inp.read().decode('utf-16')
sentences = texts.split('\r\n')  # 根据换行切分


# 将不规范的内容（如每行的开头）去掉
def clean(s):
    if u'“/s' not in s:  # 句子中间的引号不应去掉
        return s.replace(u'“ ', '')
    elif u'”/s' not in s:
        return s.replace(u'” ', '')
    elif u'‘/s' not in s:
        return s.replace(u'‘', '')
    elif u'’/s' not in s:
        return s.replace(u'’', '')
    else:
        return s

texts = u''.join(map(clean, sentences))  # 把所有的词拼接起来

# print('Length of texts is %d' % len(texts))
# print('Example of texts: \n', texts[:300])

sentence = re.split(u'[，。！？、‘’“”：（）—《》]', texts)
# print('Sentences number:', len(sentence))
# print('Sentence Example:\n', sentence[2])

#############################为每个字添加标签##############
sentences=[]
# f = open('E:\\pyCode\\Bi-directional_LSTM\\a.txt','w')
for sentenc in sentence:#给每个字添加标签
    a=sentenc.split()
    for index in range(len(a)):
        if (len(a[index]) == 1):
          a[index] += '/s  '
        elif (len(a[index]) == 2):
            a[index] = a[index][:1] + '/b  ' + a[index][1:] + '/e  '
        elif (len(a[index]) == 3):
            a[index] = a[index][:1] + '/b  ' + a[index][1:2] + '/m  '+a[index][2:]+'/e  '
        elif (len(a[index]) == 4):
            a[index] = a[index][:1] + '/b  ' + a[index][1:2] + '/m  '+a[index][2:3]+'/m  '+a[index][3:]+'/e  '
        elif (len(a[index]) == 5):
            a[index] = a[index][:1] + '/b  ' + a[index][1:2] + '/m  '+a[index][2:3]+'/m  ' + a[index][3:4] + '/m  ' + a[index][4:] + '/e  '
        elif (len(a[index]) == 6):
            a[index] = a[index][:1] + '/b  ' + a[index][1:2] + '/m  '+a[index][2:3]+'/m  '+a[index][3:4]+'/m  '+a[index][4:5]+'/m  '+a[index][5:]+'/e  '
        elif (len(a[index]) == 7):
            a[index] = a[index][:1] + '/b  ' + a[index][1:2] + '/m  '+a[index][2:3]+'/m  '+a[index][3:4]+'/m  '+a[index][4:5]+'/m  '+a[index][5:6]+'/m  '+a[index][6:]+'/e  '
        elif (len(a[index]) == 8):
            a[index] = a[index][:1] + '/b  ' + a[index][1:2] + '/m  '+a[index][2:3]+'/m  '+a[index][3:4]+'/m  '+a[index][4:5]+'/m  '+a[index][5:6]+'/m  '+a[index][6:7]+'/m  '+a[index][7:]+'/e  '
    s=u''.join(a)
    # f.write(sentences+'\n')
    sentences.append(s)
    # print(sentences)

########################
def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags # 所有的字和tag分别存为 data / label
    return None
datas = list()
labels = list()
# print('Start creating words and tags data ...')
for sentence in tqdm(iter(sentences)):
    result = get_Xy(sentence)
    if result:
        datas.append(result[0])
        labels.append(result[1])

import pickle
from sklearn.model_selection import train_test_split
# import numpy as np

with open('data/data.pkl', 'rb') as inp:
    X = pickle.load(inp)
    y = pickle.load(inp)
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)

# 划分测试集/训练集/验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,  test_size=0.2, random_state=42)
# print('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(
#     X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape))


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


# print('Creating the data generator ...')
data_train = BatchGenerator(X_train, y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)
# print('Finished creating the data generator.')

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
max_epoch = 5
max_max_epoch = 10
timestep_size = max_len = 32  # 句子长度
vocab_size = 7010  # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64  # 字向量长度
class_num = 5
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

    # 下面两部分是等价的
    # **************************************************************
    # ** 把 inputs 处理成 rnn.static_bidirectional_rnn 的要求形式
    # ** 文档说明
    # inputs: A length T list of inputs, each a tensor of shape
    # [batch_size, input_size], or a nested tuple of such elements.
    # *************************************************************
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # inputs.shape = [batchsize, timestep_size, embedding_size]  ->  timestep_size tensor, each_tensor.shape = [batchsize, embedding_size]
    # inputs = tf.unstack(inputs, timestep_size, 1)
    # ** 3.bi-lstm 计算（tf封装）  一般采用下面 static_bidirectional_rnn 函数调用。
    #   但是为了理解计算的细节，所以把后面的这段代码进行展开自己实现了一遍。
    #     try:
    #         outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
    #                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
    #     except Exception: # Old TensorFlow version only returns outputs not states
    #         outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs,
    #                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
    #     output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
    # ***********************************************************

    # ***********************************************************
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
    wordNum = tf.placeholder(tf.int32, name='wordNum')
bilstm_output = bi_lstm(X_inputs)

with tf.variable_scope('outputs'):
    softmax_w = weight_variable([hidden_size * 2, class_num])
    softmax_b = bias_variable([class_num])
    y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b

# adding extra statistics to monitor
# y_inputs.shape = [batch_size, timestep_size]
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy=(tf.reduce_sum(tf.cast(correct_prediction,tf.int32))+wordNum-16000)/wordNum
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(y_inputs, [-1]), logits=y_pred))

# ***** 优化求解 *******
tvars = tf.trainable_variables()  # 获取模型的所有参数
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # 优化器

# 梯度下降计算
train_op = optimizer.apply_gradients(zip(grads, tvars),
                                     global_step=tf.train.get_or_create_global_step())
# print('Finished creating the bi-lstm model.')
saver = tf.train.Saver()
best_model_path = 'ckpt/bi-lstm.ckpt-6'
saver.restore(sess, best_model_path)
X_tt, y_tt = data_train.next_batch(2)
# print('X_tt.shape=', X_tt.shape, 'y_tt.shape=', y_tt.shape)
# print('X_tt = ', X_tt)
# print('y_tt = ', y_tt)

# 利用 labels（即状态序列）来统计转移概率
# 因为状态数比较少，这里用 dict={'I_tI_{t+1}'：p} 来实现
# A统计状态转移的频数
A = {
    'sb': 0,
    'ss': 0,
    'be': 0,
    'bm': 0,
    'me': 0,
    'mm': 0,
    'eb': 0,
    'es': 0
}

# zy 表示转移概率矩阵
zy = dict()
for label in labels:
    for t in range(len(label) - 1):
        key = label[t] + label[t + 1]
        A[key] += 1.0

zy['sb'] = A['sb'] / (A['sb'] + A['ss'])
zy['ss'] = 1.0 - zy['sb']
zy['be'] = A['be'] / (A['be'] + A['bm'])
zy['bm'] = 1.0 - zy['be']
zy['me'] = A['me'] / (A['me'] + A['mm'])
zy['mm'] = 1.0 - zy['me']
zy['eb'] = A['eb'] / (A['eb'] + A['es'])
zy['es'] = 1.0 - zy['eb']
keys = sorted(zy.keys())
print('the transition probability: ')
for key in keys:
    print(    key, zy[key])
print(zy)
zy = {i: np.log(zy[i]) for i in zy.keys()}
print(zy)

def viterbi(nodes):
    """
    维特比译码：除了第一层以外，每一层有4个节点。
    计算当前层（第一层不需要计算）四个节点的最短路径：
       对于本层的每一个节点，计算出路径来自上一层的各个节点的新的路径长度（概率）。保留最大值（最短路径）。
       上一层每个节点的路径保存在 paths 中。计算本层的时候，先用paths_ 暂存，然后把本层的最大路径保存到 paths 中。
       paths 采用字典的形式保存（路径：路径长度）。
       一直计算到最后一层，得到四条路径，将长度最短（概率值最大的路径返回）
    """
    paths = {'b': nodes[0]['b'], 's':nodes[0]['s']} # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        print(layer)
        paths_ = paths.copy()  # 先保存上一层的路径
        print(paths_)
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path
        for node_now in nodes[layer].keys():      # nodes[layer] {'s': 6.416104, 'b': 1.4611748, 'm': -2.1077693, 'e': 1.5862198}    key为s b m e.
            print(nodes[layer])
            print(node_now)
            # 对于本层的每个节点，找出最短路径
            sub_paths = {}
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys(): # 若转移概率不为 0
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]   # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）


def text2ids(text):
    """把字片段text转为 ids."""
    words = list(text)
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        print(u'输出片段超过%d部分无法处理' % (max_len))
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    ids = np.asarray(ids).reshape([-1, max_len])
    return ids


def simple_cut(text):
    """对一个片段text（标点符号把句子划分为多个片段）进行预测。"""
    if text:
        text_len = len(text)
        X_batch = text2ids(text)  # 这里每个 batch 是一个样本
        fetches = [y_pred]
        feed_dict = {X_inputs:X_batch, lr:1.0, batch_size:1, keep_prob:1.0}
        _y_pred = sess.run(fetches, feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
        nodes = [dict(zip(['s','b','m','e'], each[1:])) for each in _y_pred]
        tags = viterbi(nodes)
        words = []
        for i in range(len(text)):
            if tags[i] in ['s', 'b']:
                words.append(text[i])
            else:
                words[-1] += text[i]
        return words
    else:
        return []


def cut_word(sentence):
    """首先将一个sentence根据标点和英文符号/字符串划分成多个片段text，然后对每一个片段分词。"""
    not_cuts = re.compile(u'([0-9\da-zA-Z ]+)|[。，、？！.\.\?,!]')
    result = []
    start = 0
    for seg_sign in not_cuts.finditer(sentence):
        result.extend(simple_cut(sentence[start:seg_sign.start()]))
        result.append(sentence[seg_sign.start():seg_sign.end()])
        start = seg_sign.end()
    result.extend(simple_cut(sentence[start:]))
    return result


sentence = "你看我盡節存忠立功勛，單注著楚霸王大軍盡。" #你  看  我  盡  節  存  忠  立  功勛  ，單  注  著  楚霸王  大軍  盡  。
result = cut_word(sentence)
rss = ''
for each in result:
    rss = rss + each + ' / '
print(rss)
