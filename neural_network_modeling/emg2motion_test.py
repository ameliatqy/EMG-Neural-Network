import tensorflow as tf
import numpy as np
import os

from matplotlib import pyplot
import math

test_features = np.genfromtxt('data/test_features.csv', delimiter=',')
test_label = np.genfromtxt('data/test_label.csv', delimiter=',')
test_angle = np.genfromtxt('data/test_angle.csv', delimiter=',')

x = test_features[5]
y = test_label[5]
theta = test_angle[5]

no_data = 50

test_x_placeholder = tf.placeholder(tf.float64, [1, no_data])

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	weights_net_1 = tf.convert_to_tensor(np.load('model/weights_1.npy'), dtype=tf.float64)
	bias_net_1 = tf.convert_to_tensor(np.load('model/bias_1.npy'), dtype=tf.float64)
	weights_y_ = tf.convert_to_tensor(np.load('model/weights_2.npy'), dtype=tf.float64)
	bias_y_ = tf.convert_to_tensor(np.load('model/bias_2.npy'), dtype=tf.float64)


	x = np.reshape(x,(1,no_data))
	hidden_out = tf.add(tf.matmul(x, weights_net_1), bias_net_1)
	y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out, weights_y_), bias_y_))

	loss = tf.losses.mean_squared_error(y_, tf.reshape(y, [1,no_data])) 


	y_degrees_calc = tf.add(tf.multiply(y_, tf.subtract(tf.cast(theta[1], tf.float64), tf.cast(theta[0], tf.float64))), tf.cast(theta[0], tf.float64));
	y_degrees = sess.run(y_degrees_calc)

	for i in range(0, y_degrees[0].shape[0]):
	 	pyplot.polar([0, theta[1]-y_degrees[0][i]], [0, 2])
	 	pyplot.polar([0, theta[1]], [0, 2])
	 	filename = 'predicted_img/%i.png'%(i)
	 	pyplot.savefig(filename)
	 	pyplot.gcf().clear()



