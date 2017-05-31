#! -*- coding:utf-8 -*-

#%pylab --no-import-all
#%matplotlib inline
import numpy as np
import pandas as pd
import dicom
import os
import math
import random
import scipy.ndimage
import tensorflow as tf
import time


class CT_Image(object):
    
    train_features = None
    input_folder = None
    depth, row, col = 0, 512, 512
    zoomrate = 1.0
    patients, train, validate, lables = [], [], [], []
	
    def __init__(self, input_folder, lable_file, zoomrate=1.0):
        self.input_folder = input_folder
        self.lables = pd.read_csv(lable_file, index_col='id')
        self.get_valid_patients(os.listdir(input_folder))
        self.train = random.sample(self.patients, int(len(self.patients)*0.8))
        self.validate = list(set(self.patients)-set(self.train))
	self.zoomrate = zoomrate
	self.depth, self.row, self.col = self.get_max_imageshape()
        self.depth = int(self.depth*self.zoomrate)
        self.row = int(self.row*self.zoomrate)
        self.col = int(self.col*self.zoomrate)

    def get_valid_patients(self, patient_list):
        for patient in patient_list:
            try:
                self.lables.loc[[patient]]
                self.patients.append(patient)
            except Exception:
                pass
		
    def get_3d_data(self, path):
	slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
	slices.sort(key = lambda x: int(x.InstanceNumber))
	return np.stack([s.pixel_array for s in slices])
	
	# Loads, resizes and processes the image
    def process_image(self, patient):
        img = self.get_3d_data(self.input_folder+patient)
        img[img == -2000] = 0
        img = scipy.ndimage.zoom(img.astype(np.float), self.zoomrate)
        img_std = np.std(img)
        img_avg = np.average(img)
        return np.clip((img-img_avg+img_std)/(img_std*2), 0, 1).astype(np.float32)
	
    def padding_image(self, patient):
	img = self.process_image(patient)
        return np.concatenate([img, np.zeros([self.depth-img.shape[0], self.row, self.col], np.float32)])
	
    def get_max_imageshape(self):
	maxdepth, maxrow, maxcol = 0, 512, 512
	for p in self.patients:
            slices = os.listdir(self.input_folder+p)
	    #firstimg = dicom.read_file(self.input_folder+p+'/'+slices[0]).pixel_array
	    maxdepth = max(maxdepth, len(slices))
	    #maxrow = max(maxrow, firstimg.shape[0])
	    #maxcol = max(maxcol, firstimg.shape[1])
	    maxdepth = math.ceil(maxdepth*1.0/4)*4
	return int(maxdepth), maxrow, maxcol

    def get_patient_lable(self, patient_list):
        train_lables = self.lables.loc[patient_list].cancer.astype(np.float32).as_matrix()
        train_lables = np.array([[lable, abs(lable-1)] for lable in train_lables]).astype(np.float32)
        return train_lables.reshape([len(train_lables), 2])
	
    def next_batch(self, num):
	if num > len(self.train):
            raise Exception('parameter num should be small then %s' % len(self.indexs)+1)
	train_list = random.sample(self.train, num)
        batch_lables = self.get_patient_lable(train_list)
        batch_train = [self.padding_image(patient) for patient in train_list]
        return batch_train, batch_lables

    def validate_batch(self):
        validate_lables = self.get_patient_lable(self.validate)
        validate_data = [self.padding_image(data) for data in self.validate]
        return validate_data, validate_lables


class CNN_3D(object):
	
    CTobj = None
	
    def __init__(self, ctobj):
	self.CTobj = ctobj
		
    def run(self):
		
	# initialize input layer
        depth, row, col = self.CTobj.depth, self.CTobj.row, self.CTobj.col
        with tf.name_scope('input'):
	    xs = tf.placeholder(tf.float32, [None, depth, row, col], name='x_input')
	    ys = tf.placeholder(tf.float32, [None, 2], name='y_input')
	    x_image = tf.reshape(xs, [-1, depth, row, col, 1])
		
	# add two convolation layer
        with tf.name_scope('convolation'):
	    h_pool1 = self.add_conv_layer(x_image, 1, 10, 1)
	    h_pool2 = self.add_conv_layer(h_pool1, 10, 20, 2)
		
	# compute some variable for fool connection layer
	depth, row, col = depth/4, row/4, col/4
	h_pool2_flat = tf.reshape(h_pool2, [-1, depth*row*col*20])
		
	# add two fool connection layer
        with tf.name_scope('fullconnection_1'):
	    W_fc1 = self.weight_variable([depth*row*col*20, 50])
	    b_fc1 = self.bias_variable([50])
	    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	    keep_prob = tf.placeholder(tf.float32)
	    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        with tf.name_scope('fullconnection_2'):
	    W_fc2 = self.weight_variable([50, 2])
	    b_fc2 = self.bias_variable([2])
        with tf.name_scope('output'):
	    #prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	    prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		
	# get the error between prediction and real data
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, ys))
	train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
        # save 3d cnn lung detect model
        saver = tf.train.Saver()

	# train graph
	with tf.Session() as sess:
            writer = tf.train.SummaryWriter("../log/", sess.graph)
            sess.run(tf.global_variables_initializer())

            """
            # train 3d cnn model
	    for step in range(200):
		batch_xs, batch_ys = self.CTobj.next_batch(10)
                feed_dict = {
                    xs: batch_xs,
                    ys: batch_ys,
                    keep_prob: 0.5
                }
                #print('ys: ', sess.run(ys, feed_dict=feed_dict))
                #print('prediction: ', sess.run(prediction, feed_dict=feed_dict))
                if step % 10 == 0:
                    print('step : %s' % step)
                    print('accuracy: ', sess.run(accuracy, feed_dict=feed_dict))
		sess.run(train_step, feed_dict=feed_dict)
            """

            print('finish train')
            #save_path = saver.save(sess, '../model/3d_cnn_lung_detect.ckpt')
            saver.restore(sess, '../model/3d_cnn_lung_detect.ckpt')

            # validate and save the 3d cnn model
	    validate_xs, validate_ys = self.CTobj.validate_batch()
            feed_dict = {
                xs: validate_xs,
                ys: validate_ys,
                keep_prob: 0.5
            }
            #print('ys: ', sess.run(ys, feed_dict=feed_dict))
            #print('prediction: ', sess.run(prediction, feed_dict=feed_dict))
            print('validate accuracy: ', sess.run(accuracy, feed_dict=feed_dict))
	    sess.run(train_step, feed_dict=feed_dict)
	
    def weight_variable(self, shape):
        with tf.name_scope('weight'):
            initial = tf.truncated_normal(shape, stddev=0.1)
	    return tf.Variable(initial)

    def bias_variable(self, shape):
        with tf.name_scope('bias'):
	    initial = tf.constant(0.1, shape=shape)
	    return tf.Variable(initial)

    def conv3d(self, x, W):
	# stride [1, depth_movement, x_movement, y_movement, 1]
	# Must have strides[0] = strides[4] = 1
	return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def max_pool_2x2x2(self, x):
	# stride [1, depth_movement, x_movement, y_movement, 1]
	return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
	
    def add_conv_layer(self, inputs, in_size, out_size, layer):
        with tf.name_scope('layer%s' % layer):
	    W_conv = self.weight_variable([5, 5, 5, in_size, out_size])
	    b_conv = self.bias_variable([out_size])
	    h_conv = tf.nn.relu(self.conv3d(inputs, W_conv) + b_conv)
	    h_pool = self.max_pool_2x2x2(h_conv)
	    return h_pool


if __name__ == '__main__':
    lable_path = '../input/stage1_labels.csv'
    images_path = '../input/stage1/'
    #images_path = '/media/wbsong/0163e9e9-9024-43fd-9087-a9764babafb3/lung_ct/stage1/'
    ctobj = CT_Image(images_path, lable_path, zoomrate=0.25)
    cnnobj = CNN_3D(ctobj)
    cnnobj.run()
