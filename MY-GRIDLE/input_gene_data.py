import numpy as np 
import tensorflow as tf
# tf_l_str = []
# with open('file_name') as f_tf:
# 	for line in f_tf.readlines():
# 		tf_l_str.append(line.split())
# gene_name = tf_l_str[0]
# expression = np.array(tf_l_str[1:])
# expression = expression.astype(float32)

class input(object):
	"""docstring for input"""
	def __init__(self,file_name, target_index):
		tf_l_str = []
		gene_name = []
		with open(file_name) as f_tf:
			for line in f_tf.readlines():
				str_split = line.split()
				if len(str_split) > 2:
					gene_name.append(str_split[0])
					tf_l_str.append(str_split[1:])
		
		ex = np.array(tf_l_str)
		# print(ex)
		expression = ex.astype(float)


		if target_index > len(gene_name):
			raise ValueError('-----------------')


		(num_gene, num_sample) = expression.shape
		self.gene_name = gene_name

		self.target = expression[target_index,:]
		self.expression = np.delete(expression,target_index,0)

		self.target_index = target_index
		self.num_gene = num_gene
		self.num_sample = num_sample

	def next_batch(self, batch_size):
		perm = np.arange(self.num_sample)
		np.random.shuffle(perm)

		batch_expression = self.expression[:,perm[:batch_size]]
		batch_expression = batch_expression.T

		batch_target = np.array([self.target[perm[:batch_size]]])


		return batch_expression, batch_target.T

	def get_target(self):
		return self.gene_name[self.target_index]


	def get_regulator(self):
		l = self.gene_name[:self.target_index] + self.gene_name[self.target_index+1:] 
		return np.array(l)

if __name__ == '__main__':
	for i in range(8):
		print('---------------------------')
		s = input('data.txt',i)

		# x,y = s.next_batch(30)
		# print(x.shape)
		# print(y)
		print(s.get_target())
		r = s.get_regulator()
		print(r)


		
		