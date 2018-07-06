

import tensorflow as tf
import numpy as np

#[fur, wings]
x_data = np.array([
    [0,0],
    [1,0],
    [1,1],
    [0,0],
    [0,0],
    [0,1],
    [1,1],
    [1,0]
])

#[etc, mammal, bird]
y_data = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1],
    [0,0,1],
    [0,1,0]
])

#number of classes, neuron
n_classes = 3
n_neuron = 10

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 3])

#[2, 10] -> [input, neuron]
W1 = tf.Variable(tf.random_normal([2, n_neuron]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([n_neuron]), name = 'bias1')
W2 = tf.Variable(tf.random_normal([n_neuron, n_classes]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([n_classes]), name = 'bias2')

#set two layers
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)
L2 = tf.add(tf.matmul(L1, W2), b2)

model = tf.nn.softmax(L2)

#Cross entropy cost/Loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

train_optimizer = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())



for step in range(100):
    sess.run(train_optimizer, feed_dict = {X: x_data, Y: y_data})

    if( (step+1)%10==0 ):
        print(step+1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

#show reverse result
prediction = tf.argmax(model, axis = 1)
target = tf.argmax(Y, axis=1)
print('predicton value: ', sess.run(prediction, feed_dict={X: x_data}))
print('real value: ', sess.run(target, feed_dict={Y: y_data}))

#print accuracy
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('accuracy: %.2f' % sess.run(accuracy * 100, feed_dict = {X: x_data, Y: y_data}))

'''
between 50 to 100 accuracy

'''