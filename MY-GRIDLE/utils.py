import tensorflow as tf
import numpy as np

def weights_biases(num_input, num_l1, num_l2, num_l3, num_out):
	weights ={
		'wl1': tf.Variable(tf.random_normal([num_input, num_l1])),
		'wl2': tf.Variable(tf.random_normal([num_l1, num_l2])),
		'wl3': tf.Variable(tf.random_normal([num_l2, num_l3])),
		'out': tf.Variable(tf.random_normal([num_l3,num_out]))
	}

	biases = {
	    'bl1': tf.Variable(tf.random_normal([num_l1])),
	    'bl2': tf.Variable(tf.random_normal([num_l2])),
	    'bl3': tf.Variable(tf.random_normal([num_l3])),
	    'out': tf.Variable(tf.random_normal([num_out]))
	}
	return weights, biases

def neural_net(x, weights, biases):
    # Hidden fully connected layer 1
    layer_1 = tf.add(tf.matmul(x, weights['wl1']), biases['bl1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['wl2']), biases['bl2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output fully connected layer 3
    layer_3 = tf.add(tf.matmul(layer_2, weights['wl3']), biases['bl3'])
    layer_3 = tf.nn.relu(layer_3)
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer