from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Load mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Start InteractiveSession to allows the C++ backend to do computations
sess = tf.InteractiveSession()

# Softmax Regression Model
    # Placeholders
xP  = tf.placeholder(tf.float32, shape=[None, 784])
y_P = tf.placeholder(tf.float32, shape=[None, 10])

    # Variables
weights = tf.Variable(tf.zeros([784,10]))
biases = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

    # Predicted Class and Loss Function
rModel = tf.matmul(xP, weights) + biases
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                              (labels=y_P, logits=rModel))

# Train the Model
    # Gradient Descent + Adding new operations to computation graph
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

    # Batch training
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={xP: batch[0], y_P: batch[1]})

# Evaluate the Model
correct_prediction = tf.equal(tf.argmax(rModel,1), tf.argmax(y_P, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={xP: mnist.test.images, y_P: mnist.test.labels}))
# 0.9197

#########################################################

# Multilayer Convolutional Network
    # Weight Initialization
def weight_variable(shape):
    initialW = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initialW)

def bias_variable(shape):
    initialB = tf.constant(.1, shape=shape)
    return tf.Variable(initialB)

    # Convolution and Pooling
def conv2d(x, weight):
    return tf.nn.conv2d(x, weight, strides=[1,1,1,1], padding='SAME')

def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # First Convolutional Layer
weightConv1 = weight_variable([5,5,1,32])
biasConv1 = bias_variable([32])
x_image = tf.reshape(xP, [-1,28,28,1])
hConv1 = tf.nn.relu(conv2d(x_image, weightConv1) + biasConv1)
hPool1 = maxPool2x2(hConv1)

    # Second Convolutional Layer
weightConv2 = weight_variable([5,5,32,64])
biasConv2 = bias_variable([64])
hConv2 = tf.nn.relu(conv2d(hPool1, weightConv2) + biasConv2)
hPool2 = maxPool2x2(hConv2)

    # Densely Connected Layer
weightFC1 = weight_variable([7*7*64, 1024])
biasFC1 = bias_variable([1024])
hPool2_flat = tf.reshape(hPool2, [-1, 7*7*64])
hFC1 = tf.nn.relu(tf.matmul(hPool2_flat, weightFC1) + biasFC1)

    # Dropout
keepProbP = tf.placeholder(tf.float32)
hFC1_drop = tf.nn.dropout(hFC1, keepProbP)

    # Readout Layer
weightFC2 = weight_variable([1024,10])
biasFC2 = bias_variable([10])
yConv = tf.matmul(hFC1_drop, weightFC2) + biasFC2

    # Training and Evaluation
crossEntropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_P, logits=yConv))
train_step = tf.train.AdamOptimizer(.0001).minimize(crossEntropy)
correct_prediction = tf.equal(tf.argmax(yConv, 1), tf.argmax(y_P, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2500):
        batch = mnist.train.next_batch(50)
        if i % 125 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                xP: batch[0], y_P: batch[1], keepProbP:1.0})
            print('Step %d, Accuracy: %g' % (i, train_accuracy))
        train_step.run(feed_dict={xP: batch[0], y_P: batch[1], keepProbP: .5})
    print('Test Accuracy %g' % accuracy.eval(feed_dict={
        xP: mnist.test.images, y_P: mnist.test.labels, keepProbP:1.0}))