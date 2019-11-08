# -*- coding: utf-8 -*-
'''
<Linear Regression 의 개념>
Linear Hypothesis는 선형으로 해결되는 문제를 가설을 세워 해결하는 방법이다
=======> H(x) = Wx + b

H(x)는 예측한 값이고 y는 결과다.
이를 통해 오차를 계산할 수 있다.
=======> H(x) - y

위의 식은 결과가 음수가 될수도 있고 양수가 될 수도 있기 때문에 제곱으로 이 문제를 해결한다.
=======> (H(x) - y)**2
위 식을 이용하여 cost function을 만들 수 있다.
=======> cost(W, b) = (H(x) - y)**2 의 평균

이러한 cost 값을 최소화 하는 지점이 학습이 완료되는 부분이다.
=======> minimize W,b cost(W,b)

'''

# Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Try to find values for W and b to compute y_data = x_data * W + b
# We know that W should be 1 and b should be 0
# But let TensorFlow figure it out
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Our hypothesis XW+b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# 이 부분은 추후에 설명. 일단 cost 최소화 알고리즘이라고 알고있으면 됨.
# train이 최상단 노드라서 실행하면 다른것도 다 실행됨
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    # variable을 사용할 때는 global_variables_initializer()를 사용해야함
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

# Learns best fit W:[ 1.],  b:[ 0.]
"""
0 2.82329 [ 2.12867713] [-0.85235667]
20 0.190351 [ 1.53392804] [-1.05059612]
40 0.151357 [ 1.45725465] [-1.02391243]
...
1960 1.46397e-05 [ 1.004444] [-0.01010205]
1980 1.32962e-05 [ 1.00423515] [-0.00962736]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
"""
