#!/usr/bin/python2
from __future__ import division, print_function
import cv2
import numpy as np
import tensorflow as tf
import glob
import re
import random

# Parameters
learning_rate = .45
training_iters = 20000
batch_size = 120
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)
dropout = 0.8 # Dropout, probability to keep units

class ConvNet(object):
    def __init__(self, session, n_input, n_classes):
        self.n_input = n_input
        self.n_classes = n_classes
        self.session = session

        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        self.weights = {
                # 5x5 conv, 1 input, 32 outputs
                'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                # 5x5 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                # fully connected, 7*7*64 inputs, 1024 outputs
                'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
                # 1024 inputs, 10 outputs (class prediction)
                'out': tf.Variable(tf.random_normal([1024, n_classes]))
                }

        self.biases = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([64])),
                'bd1': tf.Variable(tf.random_normal([1024])),
                'out': tf.Variable(tf.random_normal([n_classes]))
                }

        x_reshape = tf.reshape(self.x, shape=[-1,28,28,1])
        conv1 = conv2d(x_reshape,self.weights['wc1'],self.biases['bc1'])
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        self.pred = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        self.cost = tf.reduce_mean(tf.square(self.pred - self.y))
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_pred = tf.abs(self.pred-self.y)
        self.accuracy = tf.reduce_mean(correct_pred)

        self.session.run(tf.initialize_all_variables())

    def train(self, x, y, verbose=False):
        feed_dict = {self.x : x, self.y : y, self.keep_prob : dropout}

        if verbose:
            _, loss, acc = self.session.run([self.optimize, self.cost, self.accuracy], feed_dict=feed_dict)
            return loss, acc
        else:
            self.session.run(self.optimize, feed_dict=feed_dict)
    def predict(self, x):
        return self.session.run(self.pred, feed_dict={self.x : x, self.keep_prob: 1.})

    def get_accuracy(self, x, y):
        tacc_s = self.session.run(self.accuracy, feed_dict={self.x: x, self.y: y, self.keep_prob: 1.})
        return tacc_s
        
#
## tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input])
#y = tf.placeholder(tf.float32)
#keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
#
## Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
#
#
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
            padding='SAME')
#
#
#    # Create model
#def conv_net(x, weights, biases, dropout):
#    # Reshape input picture
#    x = tf.reshape(x, shape=[-1, 28, 28, 1])
#
#    # Convolution Layer
#    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
#    # Max Pooling (down-sampling)
#    conv1 = maxpool2d(conv1, k=2)
#
#    # Convolution Layer
#    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
#    # Max Pooling (down-sampling)
#    conv2 = maxpool2d(conv2, k=2)
#
#    # Fully connected layer
#    # Reshape conv2 output to fit fully connected layer input
#    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
#    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#    fc1 = tf.nn.relu(fc1)
#    # Apply Dropout
#    fc1 = tf.nn.dropout(fc1, dropout)
#
#    # Output, class prediction
#    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
#    return out
#
## Store layers weight & bias
#weights = {
#        # 5x5 conv, 1 input, 32 outputs
#        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
#        # 5x5 conv, 32 inputs, 64 outputs
#        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
#        # fully connected, 7*7*64 inputs, 1024 outputs
#        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
#        # 1024 inputs, 10 outputs (class prediction)
#        'out': tf.Variable(tf.random_normal([1024, n_classes]))
#        }
#
#
#biases = {
#        'bc1': tf.Variable(tf.random_normal([32])),
#        'bc2': tf.Variable(tf.random_normal([64])),
#        'bd1': tf.Variable(tf.random_normal([1024])),
#        'out': tf.Variable(tf.random_normal([n_classes]))
#        }
#
## Construct model
#pred = conv_net(x, weights, biases, keep_prob)
#
## Define loss and optimizer
##cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#cost = tf.reduce_mean(tf.square(pred-y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
## Evaluate model
#correct_pred = tf.abs(pred-y)
#accuracy = tf.reduce_mean(correct_pred)
#
## Initializing the variables
#init = tf.initialize_all_variables()
#tacc = []
#losses  = []
## Launch the graph
#
#if __name__ == "__main__":
#    with tf.Session() as sess:
#        sess.run(init)
#        step = 1
#        # Keep training until reach max iterations
#        while step * batch_size < training_iters:
#            batch_x, batch_y = gen_batch(train_data_angles, train_data_images, batch_size)
#            #cv2.imshow('trash', batch_x[0,:].reshape((28,28)))
#            #cv2.waitKey(0)
#            #print(batch_y)
#            # Run optimization op (backprop)
#            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                keep_prob: dropout})
#            if step % display_step == 0:
#                # Calculate batch loss and accuracy
#                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                    y: batch_y,
#                    keep_prob: 1.})
#                losses.append(loss)
#                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                        "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                        "{:.5f}".format(acc))
#                tacc.append(acc)
#            step += 1
#        print("Optimization Finished!")
#
#        # Calculate accuracy for all test images
#        img, lbls = gen_batch(test_data_angles, test_data_images, len(test_data_angles))
#        tacc_s = sess.run(accuracy, feed_dict={x: img,
#            y: lbls,
#            keep_prob: 1.})
#        tacc.append(tacc_s)
#        print("Testing Accuracy:", tacc_s)
#
