

import tensorflow as tf
import random
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
#from cell import ConvLSTMCell
timesteps=28
batch_size=128
total_step=10000
class Minist(object):
    def __init__(self, timesteps=0, batch_size=0,total_step=0, learning_rate=1):
        self.timesteps = timesteps
        self.num_input = 28
        self.num_class = 10
        self.batch_size = batch_size
        self.total_step = total_step
        self.learning_rate =learning_rate






def set_archit():
    my_list = ['conv', 'lstm', 'pool', 'fc']
    new_list=[]
    for i in range(8):
        secure_random = random.SystemRandom()
        x=secure_random.choice(my_list)
        if (i==0)and (x=='pool'):
            x = secure_random.choice(my_list)
        if (i==7)and (x!='fc'):
            x='fc'
        if (x=='pool')and (i>0):
            if (new_list[i-1]=='lstm'):
                x='lstm'

        print (x)
        new_list.append(x)
    return new_list

def conv_net(x,  reuse, nb_filter, size_kernel):
    #with tf.variable_scope('ConvNet', reuse=reuse):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(x, nb_filter, size_kernel, activation=tf.nn.relu)

    return conv1

def conv_nethidden(lastlayer,seq_len,  nb_filter, size_kernel,LSTM=True):
   # with tf.variable_scope('ConvNetH', reuse=reuse):
    if LSTM:
        size=lastlayer.get_shape().as_list()
        print(size[1])
        lastlayer=tf.reshape(lastlayer,shape=[-1,seq_len,int(size[1]/seq_len),1])
        conv1 = tf.layers.conv2d(lastlayer, nb_filter, size_kernel, activation=tf.nn.relu)
    else:

        conv1 = tf.layers.conv2d(lastlayer, nb_filter, size_kernel, activation=tf.nn.relu)
    return conv1

def LSTMlayer(x,seq_len,lstm_size,_weights,_biases):
    x = tf.unstack(x, seq_len, 1)
    print(len(x))
    lstm = rnn.BasicLSTMCell(lstm_size,forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm, x, dtype=tf.float32)
    print(len(outputs))
    transformed_outputs = [tf.matmul(output, _weights['out']) + _biases['out'] for output in outputs]
    final= tf.concat(axis=1, values=transformed_outputs)
    return  outputs, states,final


def from_conv_TO_lstm(net,lstm_size):
    #nn=tf.reshape(net,[-1,8,128])
    print ('net shape',net.get_shape())
    x=net.get_shape().as_list()
    print(x)
    r=int(x[1]/lstm_size)
    nn=tf.reshape(net,[-1,r,lstm_size])
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    outputs, states = tf.nn.dynamic_rnn(lstm, nn, dtype=tf.float32)
    val = tf.transpose(outputs, [1, 0, 2])
    lstm_last_output = val[-1]
    return outputs, states,lstm_last_output
#timesteps,num_hidden
def LSTM_conv(X,weights,biases,seq_len,lstm_size,nb_filter,size_kernel):
    x, y, z = LSTMlayer(X, seq_len=seq_len, lstm_size=lstm_size,_weights=weights,_biases=biases)
    conv = conv_nethidden(lastlayer=z,seq_len=seq_len, nb_filter=nb_filter, size_kernel=size_kernel, LSTM=True)

    return conv


def CONV_lstm(X,timesteps, weights,biases,nb_filter,size_kernel,lstm_size):
    #transition between CONV- pool-dense--lstm
    conv1=conv_net(X,reuse=False,nb_filter=nb_filter,size_kernel=size_kernel)
    pool = tf.layers.max_pooling2d(conv1, 2, 2)
    flat = tf.contrib.layers.flatten(pool)
    dense1 = tf.layers.dense(inputs=flat, units=1024)
    outputs,states,lstm_last_output=from_conv_TO_lstm(dense1,lstm_size=lstm_size)
    final = tf.matmul(lstm_last_output, weights['out']) + biases['out']
    return outputs,states, final


#mnit=Minist(timesteps,batch_size,total_step,learning_rate=0.001)


def main(total_step=10000 ,batch_size=128,timesteps=28,lstm_size=128,nb_filter=32,size_kernel=[5,5]):
    mnit=Minist(timesteps,batch_size,total_step,learning_rate=0.001)
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    X = tf.placeholder(tf.float32, [None, mnit.timesteps, mnit.num_input])
    Y = tf.placeholder(tf.float32, [None, mnit.num_class])
    list_archi = ['conv', 'conv', 'pool', 'fc', 'fc']
    if len(list_archi)>1:
        num_hidden=len(list_archi)-1
    else:
        num_hidden=0
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, mnit.num_class]))}
    biases = {
        'out': tf.Variable(tf.random_normal([mnit.num_class]))}
    #print(mnit.timesteps)
    #print( mnit.num_input)
    #list_archi=set_archit()
    #list_archi=['conv','lstm']
    list_archi=['conv', 'conv', 'pool', 'fc', 'fc']

    if list_archi==['conv', 'conv', 'pool', 'fc', 'fc']:

        conv1=conv_net(X, reuse=False, nb_filter=24, size_kernel=[5,5])
        size=conv1.get_shape().as_list()
        conv2=conv_nethidden(conv1, seq_len=timesteps, nb_filter=32, size_kernel=[3,3], LSTM=False)
        pool = tf.layers.max_pooling2d(conv2, 2, 2)
        print('pool', pool.get_shape().as_list())
        flat = tf.contrib.layers.flatten(pool)
        logits = tf.layers.dense(inputs=flat, units=1024)
        logits2 = tf.layers.dense(inputs=logits, units=10)

    if list_archi==['lstm', 'conv']:
    #première LSTM-CONV
        conv = LSTM_conv(X, weights, biases, timesteps, lstm_size, nb_filter, size_kernel)
        pool = tf.layers.max_pooling2d(conv, 2, 2)
        flat = tf.contrib.layers.flatten(pool)#essential to move to fully connect
        logits = tf.layers.dense(inputs=flat, units=1024)
        logits2 = tf.layers.dense(inputs=logits, units=10)
    if list_archi==['conv','lstm']:
    #deuxième CONV_LSTM
        x,y,logits2=CONV_lstm(X, timesteps, weights, biases, nb_filter, size_kernel,lstm_size)
    prediction = tf.nn.softmax(logits2)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(1, total_step + 1):
            batch_x, batch_y = mnist.train.next_batch(mnit.batch_size)
            batch_x = batch_x.reshape((mnit.batch_size,mnit.timesteps, mnit.num_input))
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % 200 == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
main()