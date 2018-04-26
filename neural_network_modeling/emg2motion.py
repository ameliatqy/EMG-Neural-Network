import tensorflow as tf
import numpy as np
import os

# Optimization parameters
lrate = 0.01
epochs = 800
batch_size =5
tf.set_random_seed(24)


# Number of inputs and outputs
no_data = 50
no_inputnodes = no_data
no_outputnodes = no_data
no_hidden1nodes = 200

data_features, data_label = (np.genfromtxt('data/input_features.csv', delimiter=','), 
                    np.genfromtxt('data/input_label.csv', delimiter=','))

dataset = tf.data.Dataset.from_tensor_slices((data_features, data_label)).repeat().shuffle(buffer_size=1000).batch(batch_size)

iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()


test_features = np.genfromtxt('data/test_features.csv', delimiter=',')
test_label = np.genfromtxt('data/test_label.csv', delimiter=',')

test_x_placeholder = tf.placeholder(tf.float64, [1, no_data])
test_y_placeholder = tf.placeholder(tf.float64, [1, no_data])

# Input Layer -> Hidden Layer
W1 = tf.Variable(tf.random_normal([no_inputnodes, no_hidden1nodes], stddev=0.03,dtype=tf.float64), name='W1', dtype=tf.float64)
b1 = tf.Variable(tf.random_normal([no_hidden1nodes], dtype=tf.float64), name='b1', dtype=tf.float64)

# Hidden Layer -> Output Layer
W2 = tf.Variable(tf.random_normal([no_hidden1nodes, no_outputnodes], stddev=0.03,dtype=tf.float64), name='W2', dtype=tf.float64)
b2 = tf.Variable(tf.random_normal([no_outputnodes], dtype=tf.float64), name='b2', dtype=tf.float64)

hidden_out = tf.add(tf.matmul(x,W1), b1)

y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out, W2), b2))


loss = tf.losses.mean_squared_error(y_, y)
optimiser = tf.train.GradientDescentOptimizer(learning_rate=lrate).minimize(loss)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)
	for i in range(epochs):
		_, loss_value = sess.run([optimiser, loss])
		print("Iter: {}, Loss: {:.4f}, MSE: {:.4f}".format(i, loss_value, sess.run(loss)))
		#print("Prediction: {}, Actual: {}".format(sess.run(y_), sess.run(y)))

	test_x = np.reshape(test_features[0],(1,no_data))

	# Save Model
	np.save('model/weights_1.npy', sess.run(W1))
	np.save('model/weights_2.npy', sess.run(W2))
	np.save('model/bias_1.npy', sess.run(b1))
	np.save('model/bias_2.npy', sess.run(b2))


