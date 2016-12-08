import pandas as pd
import glob
import numpy as np
import os
import random
import cv2

class Formatter(object):
    def __init__(self,n_input,path='.'):
        self.imgs = []
        #Process Images
        for img in glob.glob(os.path.join(path, '*.jpeg')):
            n  = cv2.cvtColor(cv2.resize(cv2.imread(img), (28,28)), cv2.COLOR_BGR2GRAY)
            n = np.asarray(n)
            n = np.reshape(n, n_input)
            self.imgs.append(n) # img_array

        angle = []
        image_number = []

        #Process File for angle, here we read the text line by line and make a list
        with open(os.path.join(path, 'allinfo.txt')) as f:
            content = f.readlines()
            for i in range(len(content)):
                content[i] = content[i][:-1].split(',')
                image_number.append(float(content[i][1]))
                angle.append(float(content[i][7]))

        #Initialize arrays to unpack data file

        #Iterate through the text list and split each one by the comma separating the values. 
        #Turn the text into floats for use in the network
        #Divide both angle and image number into test and train data sets

        angle = np.atleast_2d(angle).T
        angle = 10000*angle

        ##Encode angle into 10 classes (it ranges -1 to 1)
        #for i in range(len(angle)):
        #    angle[i] = random.uniform(-1,1)
        #    angle[i] = int((angle[i]+1.0)*n_classes/2.)
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
        train_percent = .8
        train_number = int(len(test_data)*train_percent)
        train_data = np.zeros((train_number, 2))

        for i in range(train_number):
            rand = random.randrange(0,len(test_data))
            train_data[i] = test_data[rand]
            test_data = np.delete(test_data, rand, 0)

        self.test_data_images, self.test_data_angles = test_data[:,0], test_data[:,1]
        self.train_data_images, self.train_data_angles = train_data[:,0], train_data[:,1]

    def gen_batch(self,batch_size,test=False):

        if test:
            angles = self.test_data_angles
            images = self.test_data_images
        else:
            angles = self.train_data_angles
            images = self.train_data_images

        indices = random.sample(xrange(0,len(images)), batch_size)
        batch_images = []
        batch_angles = []
        for i in range(batch_size):
            batch_images.append(self.imgs[int(images[indices[i]])][:])
            batch_angles.append(angles[indices[i]])
        batch_images = np.asarray(batch_images)
        batch_angles = np.asarray(batch_angles)

        return batch_images, batch_angles

if __name__ == "__main__":
    formatter = Formatter('.')
