#!/usr/bin/python
from __future__ import print_function

# Network Stuff
import tensorflow as tf
from nnet import ConvNet
from formatter import Formatter
from matplotlib import pyplot as plt
import numpy as np

# ROS Stuff
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

# Parameters
learning_rate = .45
training_iters = 200
batch_size = 1
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)
dropout = 0.8

class img2cmd(object):
    def __init__(self, net):
        self.net = net
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image_raw",Image,self.callback)
        self.cmd_pub = rospy.Publisher('/net_cmd', Twist, queue_size=10)
        self.cmd_msg = Twist()

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        img = cv2.cvtColor(cv2.resize(cv_image, (28,28)), cv2.COLOR_BGR2GRAY)
        # Why the do you flatten it here?
        img = np.asarray(img)
        img = np.reshape(img, (1, n_input))

        cmd_ang = self.net.predict(img)
        self.cmd_msg.angular.z = cmd_ang
        self.cmd_pub.publish(self.cmd_msg)
  

def train(form, net):
    step = 1

    losses = []
    tacc = []
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = form.gen_batch(batch_size, test=False)

        # Run optimization op (backprop)
        verbose = (step % display_step == 0)
        res = net.train(batch_x, batch_y, verbose)
        if verbose:
            loss,acc = res
            losses.append(loss)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
            tacc.append(acc)
        step += 1

    print("Optimization Finished!")

    # Calculate accuracy for all test images
    img, lbls = form.gen_batch(batch_size, test=True)
    tacc_s = net.get_accuracy(img, lbls)
    tacc.append(tacc_s)
    print("Testing Accuracy:", tacc_s)

    plt.plot(losses)
    plt.plot(tacc)

def main(args):
    rospy.init_node('nnet_ai', anonymous=False)
    form = Formatter(n_input, 'data') # imgs and labels in '${PWD}/data'
    with tf.Session() as sess:
        net = ConvNet(sess, n_input, n_classes)
        # TRAIN ... 
        train(form, net)
        # NOW OUTPUT CMDs based on input camera img...
        ic = img2cmd(net)
        rospy.spin()

if __name__ == "__main__":
    main(sys.argv)
