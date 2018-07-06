

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

#number of classes
n_classes = 3

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([2, n_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([n_classes]), name = 'bias')

L1 = tf.add(tf.matmul(X, W), b)
L1 = tf.nn.relu(L1)

model = tf.nn.softmax(L1)

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
poor accuracy
'''