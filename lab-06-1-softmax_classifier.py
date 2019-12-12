# -*- coding: utf-8 -*-
# Lab 6 Softmax Classifier
"""
A B C 등급을 나눈다고 가정하자. 지난 5강에서 들었던 Sigmoid function을 이용한 Binary Classification을 복합적으로 이용하여 충분히 구현할 수 있다. A or not인 function, B or not인 function 그리고 C or not인 function을 이용하여 인풋에 따라 Binary가 아닌 Multinomial Classification을 구현할 수 있다.

위와 같이 개별적인 Binary Classification을 통해 나오는 결과값이 있다. 이것은 개별적인(A, B, C)사건에 대한 예측이다. 예를들어 결과가 A:2.0, B:1.0, C:0.1 이렇게 개별 결과가 나왔을 때, 이를 Softmax function을 이용하면 각 개별 사건의 합이 1이 되도록 만들 수 있다. A:0.7, B:0.2, C:0.4 와 같이 말이다. 이렇게 되면 이를 개별 사건의 결과라기보다는 해당 선택지의 확률로 볼 수 있게된다. 더해서 One-Hot encoding 기법을 이용하면 1.0, 0.0, 0.0의 결과를 얻을 수 있다. 가장 큰 확률을 선택하는 방법이다.

"""

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

            if step % 200 == 0:
                print(step, cost_val)

    print('--------------')
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))

'''
0 6.926112
200 0.6005015
400 0.47295815
600 0.37342924
800 0.28018373
1000 0.23280522
1200 0.21065344
1400 0.19229904
1600 0.17682323
1800 0.16359556
2000 0.15216158
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]] [1]
-------------
[[0.9311919  0.06290216 0.00590591]] [0]
-------------
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]
 [9.3119192e-01 6.2902197e-02 5.9059085e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]] [1 0 2]
'''
