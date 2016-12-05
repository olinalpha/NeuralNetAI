#!/usr/bin/python
import cv2
import numpy as np
import tensorflow as tf
import glob
import re
import random


# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 120
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



image = np.reshape(np.asarray(mnist.train.images[0]), (28,28))

#Process Images

cv_img = []
for img in glob.glob("./images/*.jpeg"):
    n  = cv2.cvtColor(cv2.resize(cv2.imread(img), (28,28)), cv2.COLOR_BGR2GRAY)
    n = np.asarray(n)
    n = np.reshape(n, n_input)
    cv_img.append(n)

#Process File for angle, here we read the text line by line and make a list
with open("./images/allinfo.txt") as f:
    content = f.readlines()

#Initialize arrays to unpack data file
angle = []
image_number = []


#Iterate through the text list and split each one by the comma separating the values. 
#Turn the text into floats for use in the network
for i in range(len(content)):
    content[i] = content[i][:-1].split(',')
    image_number.append(float(content[i][1]))
    angle.append(float(content[i][7]))

#Divide both angle and image number into test and train data sets
angle = np.atleast_2d(angle).T

#
##Encode angle into 10 classes (it ranges -1 to 1)
#for i in range(len(angle)):
#    angle[i] = random.uniform(-1,1)
#    angle[i] = int((angle[i]+1.0)*n_classes/2.)
#
#
##Create a one-hot version of angle
#angle_one_hot = np.zeros((len(angle),n_classes))
#
#for c in range(len(angle)):
#    one_hot = np.zeros(n_classes)
#    one_hot[int(angle[c])] = 1
#    angle_one_hot[c] = one_hot


image_number = np.atleast_2d(image_number).T
test_data =  np.hstack((image_number, angle))
#print test_data
train_percent = .8
train_number = int(len(test_data)*train_percent)
train_data = np.zeros((train_number, 2))
for i in range(train_number):
    rand = random.randrange(0,len(test_data))
    train_data[i] = test_data[rand]
    test_data = np.delete(test_data, rand, 0)
test_data_images = test_data[:,0]
test_data_angles = test_data[:,1]
train_data_images, train_data_angles = train_data[:,0], train_data[:,1]



def gen_batch(angles, images, batch_size, image_array=cv_img):
    indices = random.sample(xrange(0,len(images)), batch_size)
    batch_images = []
    batch_angles = []
 #   print angles
    for i in range(batch_size):
        batch_images.append(image_array[int(images[indices[i]])][:])
        batch_angles.append(angles[indices[i]])
    batch_images = np.asarray(batch_images)
    batch_angles = np.asarray(batch_angles)
    
    return batch_images, batch_angles


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
        
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = tf.reduce_mean(pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize((pred-y)**2)

# Evaluate model
correct_pred = pred
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = gen_batch(train_data_angles, train_data_images, batch_size)
        #cv2.imshow('trash', batch_x[0,:].reshape((28,28)))
        #cv2.waitKey(0)
        #print(batch_y)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for all test images
    img, lbls = gen_batch(test_data_angles, test_data_images, len(test_data_angles))
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: img,
                                      y: lbls,
                                      keep_prob: 1.})
