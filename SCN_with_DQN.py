import tensorflow as tf
import numpy as np
import gym
import random

class DQN():
	def __init__(self, sess, name,num_act, grid_size,num_state_frames):

		self.num_act = num_act # number of possible actions. Assumed action space is discrete.
		self.alpha = 0.0002  # learning rate
		self.sess = sess
		self.name = name
		self.grid_size = grid_size
		self.inputdim = [grid_size,grid_size,num_state_frames]
		self.num_state_frames = num_state_frames
		self.CNN()

	def CNN(self):
		with tf.variable_scope(self.name):
			
			# Non linear part. Normal CNN

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

			# Linear part.
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


# preprocessing : 
# rgb image to gray scale image
# change image size to specific grid size to handle it easily
class pre_processor():
	def __init__(self, sample_state, grid_size):
		with tf.variable_scope("pre_processor"):
			self.input = tf.placeholder(shape=sample_state.shape, dtype=tf.uint8)
			self.output = tf.image.rgb_to_grayscale(self.input)
			self.output = tf.image.resize_images(self.output, [grid_size, grid_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output = tf.squeeze(self.output)
	def process(self,sess,state):
		return sess.run(self.output, feed_dict = {self.input : state})


# Will use it when we need to update weights of graph that gives us target Q values. 
def update_target(major_scope_name, target_scope_name, sess):
	ori_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=major_scope_name)
	tar_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope_name)

	update_list = []
	for ori, tar in zip(ori_weights, tar_weights):
		update_list.append(tar.assign(ori.value()))

	sess.run(update_list)
	

# main part combining all above functions/classes to run the code
def main(game):

	# Hyper Parameters
	epsilon = 1
	epsilon_l_bound = 0.01
	epsilon_steps = (epsilon - epsilon_l_bound)/1000
	epochs = 10000
	game_over = False
	env = gym.make(game)
	discount_factor = 0.99
	batch_size = 32
	grid_size = 84
	num_state_frames = 4
	replay_memory_size = 50000
	target_update_steps = 500




	# pick a sample state to know the shape of input image(since different games have different image size) 
	# and possible action numbers
	sample_state = env.reset()
	num_act = env.action_space.n

	sess = tf.Session()


	# will use this object when processing images
	pre_proc = pre_processor(sample_state, grid_size)


	# separate scopes so that we can separately update weights
	major_scope_name = 'Major'
	target_scope_name = 'Target'
	major = DQN(sess, major_scope_name, num_act, grid_size, num_state_frames)
	target = DQN(sess, target_scope_name, num_act, grid_size, num_state_frames)

	init = tf.global_variables_initializer()
	sess.run(init)

	# Since both major and target networks are randomly initialized, 
	# we need to make the weights of both network the same from the first
	update_target(major_scope_name, target_scope_name, sess)
	
	cur_epoch = 0
	scores = []
	replay_memory = []

	# current is used to fill in data into replay memory
	current = 0

	for i in range(epochs):
		
		epsilon = 1
		step = 0
		cur_epoch_reward = 0

		# since we are expecting image input, which is stationary by one of itself, we need to take in several images at once
		# so that we include movement information. num_state_frames is the number of images we take in at once
		current_state = np.zeros([1,grid_size, grid_size, num_state_frames])

		# st means state of one image input. What we are going to put into the network is sum of it, current_state
		st = env.reset()
		processed_st = pre_proc.process(sess,st)
		game_over = False

		# since we do not have enough frames until we do some actions, we will temporarily use the same states 
		# to populate current_state
		for i in range (num_state_frames):
			current_state[:,:,:,i] =  processed_st


		while not game_over:
			
			# epsilon greedy policy
			if epsilon > np.random.rand(1):
				act = np.random.randint(num_act)
			else :
				act = np.argmax(major.get_q_val(current_state))

			next_frame, reward, game_over, info = env.step(act)

			cur_epoch_reward += reward
			next_state = np.zeros([1, grid_size, grid_size, num_state_frames])


			# Insert newly gained image state into next_state
			next_state[:,:,:, :num_state_frames - 1] = current_state[:,:,:, 1:]
			next_state[:,:,:, 3] = pre_proc.process(sess, next_frame)


			if current < replay_memory_size:
				replay_memory.append([current_state[0], act, reward, next_state[0], game_over])
			else:
				replay_memory[current % replay_memory_size] = [current_state[0], act, reward, next_state[0], game_over]
			
			current += 1
			
			# Randomly sample batch from replay memory to train networks
			batch = random.sample(replay_memory, min(batch_size, current))

			state_batch, act_batch, reward_batch, next_state_batch, game_over_batch = map(np.array, zip(*batch))

			q_next_val = target.get_q_val(next_state_batch)
			target_loss_args = reward_batch + (1 - game_over_batch) * discount_factor * np.amax(q_next_val, axis=1)

			# Train major nn
			major.update(state_batch, target_loss_args, act_batch)
			


			# decrease epsilon
			if epsilon >= epsilon_l_bound:
				epsilon -= epsilon_steps

			# update target network when specific amout of steps are passed
			if step % target_update_steps == 0: 
				update_target(major_scope_name, target_scope_name, sess)

			current_state = next_state
			step += 1
			
		cur_epoch += 1

		scores.append(cur_epoch_reward)
		print("epoch : {}, step : {}, reward : {}, avg_reward : {}".format(cur_epoch, step, cur_epoch_reward, np.average([scores])))