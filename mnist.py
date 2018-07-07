import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

##
## construct neuron model
##

n_classes = 10

X = tf.placeholder(tf.float32, [None, 784]) # 28 * 28 pixel
Y = tf.placeholder(tf.float32, [None, n_classes])  # 0 to 9

# 784 input datas -> 256 neurons -> 256 neurons -> 10 output datas

W1 = tf.Variable(tf.random_normal([784, 256], stddev = 0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev = 0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, n_classes], stddev = 0.01))

model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

'''
##
## train model
##
sess = tf.Session()
sess.run(tf.global_variables_initializer)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
'''