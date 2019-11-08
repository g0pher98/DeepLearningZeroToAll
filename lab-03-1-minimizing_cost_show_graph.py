# -*- coding: utf-8 -*-
'''
<Linear Regression cost함수 최소화>
H(x) = Wx + b 의 cost function은 이차함수 형태다.
여기서 최소점이 학습이 완료되는 시점이고, 이 최솟값을 찾는 알고리즘이 많다.
대표적으로 사용하는 알고리즘이 Gradient descent algorithm이다.
'''
# Lab 3 Minimizing Cost
import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Variables for plotting cost function
W_history = []
cost_history = []

# Launch the graph in a session.
with tf.Session() as sess:
    for i in range(-30, 50):
        curr_W = i * 0.1
        curr_cost = sess.run(cost, feed_dict={W: curr_W})

        W_history.append(curr_W)
        cost_history.append(curr_cost)

# Show the cost function
# cost함수 시각화,,, 구름 ide에서는 디스플레이가 없어서 안됨.
plt.plot(W_history, cost_history)
plt.show()
