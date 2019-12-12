# Lab 7 Learning rate and Evaluation

"""
<Learning rate>
GradientDescent 알고리즘에서 learning rate를 너무 크게 설정하면 overshooting이 발생하여 cost 함수 밖으로 발산된다. 반면에 너무 작게 설정되면 최적의 점에 도달하는데에 시간이 너무 오래걸린다. 즉, 이 learning rate를 얼마나 적절히 설정하느냐에 따라 학습의 효율 및 결과를 다르게 할 수 있다.

<Data preprocessig>
입력값(x)의 분포가 넓다면 학습에 문제가 생기기도 한다. 두개의 입력이 있다고 가정했을 때, x1의 분포는 1~10인데 x2의 분포가 -2000 ~ 9000일 경우 cost function이 찌그러진 형태로 분포된다. 이럴 경우에 최적화 과정에서 어느점에서 시작하느냐에 따라 학습률이 달라진다. 이를 해결하기 위해서 zero-centerd data 방식으로 분포의 중심을 0,0에 맞추는 방법이 있고, normalized data 방식으로 분포를 줄여서 보다 덜 찌그러진 형태로 변형하는 방법도 있다.

<Overfitting>
학습에서 가장 큰 문제점은 overfitting이다. 학습 데이터에 너무 치중된 학습을 하게되면 학습데이터에 있어서는 훌륭한 결과를 낼 지라도, 주어지지 않은 데이터셋에서 좋지 못한 결과를 낼 수도 있다. 결국 주어진 데이터셋에 너무 딱맞게 학습하는것도 좋지 않다. 이를 해결하기 위해서는 더 많은 데이터셋을 학습시키거나, 중복 데이터를 제거하는 방법이 있지만 이런 방식보다는 Regularization이라는 테크니컬한 방식으로 해결하는것이 좋다. 학습한 모델의 굽은 정도를 펴주는 방식이다. 너무 학습을 하게되면 굉장히 데이터에 맞게 구불구불 해지는데 이를 펴주는 것이다. cost에 Regularization strength 라는 상수를 더해 구현해줄 수 있다. 이 값은 중요도에 비례한다. 크면 그만큼 중요하다는 것이고, 작으면 무시한다는 것이다.


<Training Data>
데이터셋을 전부 그대로 학습시키는 것은 좋지 않다. 마치 똑같은 시험을 여러번 치게 만드는 것과 같다. (==암기) 그래서 70%정도는 학습용도로 이용하고, 학습이 완료되면 나머지 30%를 테스트 셋으로 확인하는것이 좋다. 더 세분화 하게되면 Training(학습 데이터셋), Validation(모의평가 데이터셋), Testing(실질 평가 데이터셋) 이렇게 여러개로 나눌 수도 있다.

<Online learning>
데이터셋이 너무 많을 경우 이 방법을 사용할 수 있다. 데이터셋을 나누어서 학습 후 그 결과를 보존한 상태에서 다음 데이터셋을 학습하는 방식이다. 계속 갱신할 수 있다는 점이 장점이다.


"""


import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# Try to change learning_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))

'''
when lr = 1.5

0 5.73203 [[-0.30548954  1.22985029 -0.66033536]
 [-4.39069986  2.29670858  2.99386835]
 [-3.34510708  2.09743214 -0.80419564]]
1 23.1494 [[ 0.06951046  0.29449689 -0.0999819 ]
 [-1.95319986 -1.63627958  4.48935604]
 [-0.90760708 -1.65020132  0.50593793]]
2 27.2798 [[ 0.44451016  0.85699677 -1.03748143]
 [ 0.48429942  0.98872018 -0.57314301]
 [ 1.52989244  1.16229868 -4.74406147]]
3 8.668 [[ 0.12396193  0.61504567 -0.47498202]
 [ 0.22003263 -0.2470119   0.9268558 ]
 [ 0.96035379  0.41933775 -3.43156195]]
4 5.77111 [[-0.9524312   1.13037777  0.08607888]
 [-3.78651619  2.26245379  2.42393875]
 [-3.07170963  3.14037919 -2.12054014]]
5 inf [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]
6 nan [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]
 ...
Prediction: [0 0 0]
Accuracy:  0.0
-------------------------------------------------
When lr = 1e-10

0 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
1 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
...
199 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
200 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
Prediction: [0 0 0]
Accuracy:  0.0
-------------------------------------------------
When lr = 0.1

0 5.73203 [[ 0.72881663  0.71536207 -1.18015325]
 [-0.57753736 -0.12988332  1.60729778]
 [ 0.48373488 -0.51433605 -2.02127004]]
1 3.318 [[ 0.66219079  0.74796319 -1.14612854]
 [-0.81948912  0.03000021  1.68936598]
 [ 0.23214608 -0.33772916 -1.94628811]]
...
199 0.672261 [[-1.15377033  0.28146935  1.13632679]
 [ 0.37484586  0.18958236  0.33544877]
 [-0.35609841 -0.43973011 -1.25604188]]
200 0.670909 [[-1.15885413  0.28058422  1.14229572]
 [ 0.37609792  0.19073224  0.33304682]
 [-0.35536593 -0.44033223 -1.2561723 ]]
Prediction: [2 2 2]
Accuracy:  1.0
'''
