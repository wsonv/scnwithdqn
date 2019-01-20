import tensorflow as tf

class DQN():
	def __init__(self, sess, name,num_act, grid_size,num_state_frames):

		self.num_act = num_act # number of possible actions. Assumed action space is discrete.
		self.alpha = 0.0002  # learning rate
		self.sess = sess
		self.name = name
		self.grid_size = grid_size
		self.inputdim = [grid_size,grid_size,num_state_frames]
		self.num_state_frames = num_state_frames
		self.build_network()

	def build_network(self):
		with tf.variable_scope(self.name):
			
			# Non linear part
			# Normal CNN

			self.X = tf.placeholder(tf.float32, shape = [None] + self.inputdim, name='input')
			self.Y = tf.placeholder(tf.float32, shape = [None], name='output')
			self.act = tf.placeholder(tf.int32, shape = [None])
			W1 = tf.Variable(tf.random_uniform([9,9,self.num_state_frames,32],-0.1,0.1))
			W2 = tf.Variable(tf.random_uniform([7,7,32,64],-0.1,0.1))
			W3 = tf.Variable(tf.random_uniform([5,5,64,64],-0.1,0.1))

			size1 = self.grid_size - 9 + 1
			size2 = size1 - 7 + 1
			size3 = size2 - 5 + 1

			# Weights of fully connected layers
			F1 = tf.Variable(tf.random_uniform([size3 * size3 * 64, 500],-0.1,0.1)) 
			F2 = tf.Variable(tf.random_uniform([500, self.num_act],-0.1,0.1)) 
			
			# biases
			b1 = tf.Variable(tf.constant(0.1, shape=[32]))
			b2 = tf.Variable(tf.constant(0.1, shape=[64]))
			b3 = tf.Variable(tf.constant(0.1, shape=[64]))
			b4 = tf.Variable(tf.constant(0.1, shape=[500]))
			

			y1 = tf.nn.relu(tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='VALID') + b1)
			y2 = tf.nn.relu(tf.nn.conv2d(y1, W2, strides=[1, 1, 1, 1], padding='VALID') + b2)
			y3 = tf.nn.relu(tf.nn.conv2d(y2, W3, strides=[1, 1, 1, 1], padding='VALID') + b3)
			
			# fully connected layers
			y3_flat = tf.reshape(y3, [-1, size3 * size3 * 64])
			y4 = tf.nn.relu(tf.matmul(y3_flat, F1) + b4)

			# Linear part
			# Set bias as 0 for simplicity
			flat_input = tf.reshape(self.X, [-1,self.grid_size * self.grid_size * self.num_state_frames])
			K = tf.Variable(tf.random_uniform([self.grid_size * self.grid_size * self.num_state_frames, self.num_act]))


			non_lin_output = tf.matmul(y4, F2)
			lin_output = tf.matmul(flat_input, K)

			# Output would be sum of non-linear and linear results.
			# By including K in network, we are expecting it to be updated with back propagation
			# Since there is no hidden layers between linear output and input, the result is still linear to input

			self.output = non_lin_output + lin_output


		act_one_hot = tf.one_hot(self.act, self.num_act, 1.0, 0.0) 
		one_hot_output = tf.reduce_sum(tf.multiply(self.output, act_one_hot), reduction_indices=1)

		loss = tf.reduce_mean(tf.square(self.Y - one_hot_output))
		optimizer = tf.train.RMSPropOptimizer(self.alpha, 0.99, 0.0, 1e-6)
		self.train = optimizer.minimize(loss)

	# getting Q value
	def get_q_val(self, state):
		return self.sess.run(self.output, feed_dict = {self.X : state})

	# train the weights
	def update(self, state_batch, target_loss_args, act_batch):
		self.sess.run(self.train, feed_dict = {self.X : state_batch, self.Y : target_loss_args, self.act : act_batch})
