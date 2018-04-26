import tensorflow as tf
import numpy as np
import os

from matplotlib import pyplot
import math

no_data = 50

init_op = tf.global_variables_initializer()



with tf.Session() as sess:

	weights_net_1 = tf.convert_to_tensor(np.load('model/weights_1.npy'), dtype=tf.float64)
	bias_net_1 = tf.convert_to_tensor(np.load('model/bias_1.npy'), dtype=tf.float64)
	weights_y_ = tf.convert_to_tensor(np.load('model/weights_2.npy'), dtype=tf.float64)
	bias_y_ = tf.convert_to_tensor(np.load('model/bias_2.npy'), dtype=tf.float64)

	filename_features = ['data/input_features.csv', 'data/test_features.csv']
	filename_label = ['data/input_label.csv', 'data/test_label.csv']

	for f in range(0,2):
		features = np.genfromtxt(filename_features[f], delimiter=',')
		label = np.genfromtxt(filename_label[f], delimiter=',')
		no_samples = features.shape[0]

		wMAPE_num = [];
		wMAPE_denom = [];

		for i in range(0,no_samples):
			x = features[i]
			y = label[i]
			x = np.reshape(x,(1,no_data))
			hidden_out = tf.add(tf.matmul(x, weights_net_1), bias_net_1)
			y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out, weights_y_), bias_y_))

			cross_entropy = tf.losses.mean_squared_error(y_, tf.reshape(y, [1,no_data])) 

			wMAPE_num.append(np.sum(sess.run(tf.abs(tf.subtract(y_, y)))[0]))
			wMAPE_denom.append(np.sum(y))
	
		accuracy = 1 - np.divide(np.sum(wMAPE_num), np.sum(wMAPE_denom))

		if f == 0:
			print 'Training Accuracy: ', accuracy
		else:
			print 'Testing Accuracy: ', accuracy

	

	
	#accuracy = tf.subtract(tf.cast(1, tf.float64), tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y_, y)),y)))
