import tensorflow as tf
import numpy as np
from input_gene_data import *
from utils import *
import argparse
import os

def read_list(list_name):
	target_list = []
	with open(list_name) as f:
		for line in f.readlines():
			list_split = line.split()
			if len(list_split) > 1:
				for i in list_split:
					target_list.append(int(i))
	print(target_list)
	return target_list


def write_file(file, target2write, regulater2write):
	file.write(str(target2write))
	file.write(' :')
	file.write(str(regulater2write))
	file.write('\n')


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_file", type=str, default='data.txt')
	parser.add_argument("-o", "--output_file", type=str, default='output.txt')
	parser.add_argument("-t", "--target_gene", type=int, default= 0)
	parser.add_argument("-b", "--bottleneck_layer", type=int, default=3)
	parser.add_argument("--target_gene_list", type=str, default=None)

	return parser.parse_args()

def Gridle():

	args = parse_args()


	learning_rate = 0.001
	num_steps = 5000
	batch_size = 60
	display_step = 500

	# the input are sampled by bootstrap sqrt(p) 
	num_input = 9 
	num_out = 1

	# tf placeholder input
	X = tf.placeholder(tf.float32,[None,num_input])
	Y = tf.placeholder(tf.float32,[None,num_out])


	weights, biases = weights_biases(num_input,3, 18, 9, num_out)
	logits = neural_net(X, weights, biases)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.pow(logits-Y,2))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()
	w_list = np.zeros(9)
	# Start training
	count = 1
	if args.target_gene_list:
		tar_list = read_list(args.target_gene_list)
		count = len(tar_list)
	else:
		tar_list = [args.target_gene]

	ffile = open(args.output_file,'w')
	while count:
		count -= 1
		train = input(args.input_file, tar_list[count])
		print('Running for target:', train.get_target())

		with tf.Session() as sess:
		    # Run the initializer
		    sess.run(init)

		    for step in range(1, num_steps+1):
		        batch_x, batch_y = train.next_batch(batch_size)

		        sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})
		        if step % display_step  == 0 or step == 1:
		            loss = sess.run(loss_op, feed_dict={X: batch_x, Y:batch_y})

		            print("Step" + str(step) + ", Minibatch loss: ", loss)
		        if num_steps-step < 1000 and (num_steps-step)%5 ==0:
		            weight_l = sess.run(weights['wl1'])
		            weight_for_gene = np.abs(np.sum(weight_l, axis=1))
		            w_list += weight_for_gene

		    print("Optimization Finished!")
		    print("Extract the weights of first layer")
		    w_list = w_list/np.max(w_list)
		    print(w_list)

		    sort = np.argsort(-w_list)
		    out_gene = train.get_regulator()[sort[:4]]
		    print(sort)
		    print(out_gene)
		    write_file(ffile, train.get_target(), out_gene)
if __name__ == '__main__':
	Gridle()

