# -*- coding: utf-8 -*-
# Lab 5 Logistic Regression Classifier
"""
learning_rate란 cost함수의 최저점을 찾기 위해 찾아가는 걸음(hop)의 크기를 말한다.

기존 linear Regression에서 발생하는 문제가 있다. 10시간 기준으로 공부시간에 따른 pass/non-pass를 결정하는 문제가 있다면 10시간 내외 데이터를 가지고 학습시켰을 때, 올바른 결과가 나올 수 있겠지만, 100시간이 넘는 데이터를 학습하게 되면 큰 오차가 발생한다. 또한 결과가 pass(1)나 non-pass(0) 둘 중 하나여야 하는데 결과 값이 큰 정수가 되어버릴 수도 있다.

그래서 이러한 문제를 해결하고자 기존에 사용하던 H(x)=Wx+b 가 아닌 새로운 형태의 0과 1 범위 내에서 유연한 함수를 찾았는데 이것이 Logistic Classification 또는 Sigmoid function 이라고 부른다. 

새롭게 찾은 함수를 g(x)라고 할 때, H(x)=g(Wx) 형태로 만들어서 sigmoid 함수를 완성시킨다.

그러나 기존의 방식으로 cost 함수를 세웠을 때, 꾸불꾸불한 형태를 띄고 있어서 어느 점에서 시작하냐에 따라 local minimum이 달라진다. 이 local minimum이 global mimimum과 같아지는 점을 찾아야 한다.

이를 해결하기 위해 cost 함수를 다른 방식으로 세울 수 있다. log를 씌워서 꼬불꼬불함을 한층 풀어주는데 식은 다음과 같다.
{
    -log(H(x))      : y=1
    -log(1 - H(x))  : y=0
}
결국 정답일 경우 cost 값은 0에 가까워지고, 오답일 경우 무한에 가까워진다. (== 학습에 적당한 cost 함수의 모습)

위 처럼 분할된 cost 함수를 합쳐서 하나의 식으로 세우면 다음과 같다.
-y*log(H(x)) - (1-y)*log(1 - H(x))

위 함수의 모습은 흡사 이전에 사용했던 linear regression의 cost 함수와 비슷한 형태를 띄기 때문에 Gradient decent algorihtm을 통해서 cost를 minimise를 할 수 있다.

"""

import tensorflow as tf
#tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
0 1.73078
200 0.571512
400 0.507414
600 0.471824
800 0.447585
...
9200 0.159066
9400 0.15656
9600 0.154132
9800 0.151778
10000 0.149496

Hypothesis:  [[ 0.03074029]
 [ 0.15884677]
 [ 0.30486736]
 [ 0.78138196]
 [ 0.93957496]
 [ 0.98016882]]
Correct (Y):  [[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  1.0
'''
