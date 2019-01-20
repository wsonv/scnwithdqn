import tensorflow as tf
import numpy as np
import gym
import random
import argparse
import DQN
import pre_processor



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
	epsilon_steps = (epsilon - epsilon_l_bound)/100000
	epochs = 10000
	game_over = False
	env = gym.make(game)
	discount_factor = 0.99
	batch_size = 32
	grid_size = 84
	num_state_frames = 4
	replay_memory_size = 100000
	target_update_steps = 5000




	# pick a sample state to know the shape of input image(since different games have different image size) 
	# and possible action numbers
	sample_state = env.reset()
	num_act = env.action_space.n

	sess = tf.Session()


	# will use this object when processing images
	pre_proc = pre_processor.pre_processor(sample_state, grid_size)


	# separate scopes so that we can separately update weights
	major_scope_name = 'Major'
	target_scope_name = 'Target'
	major = DQN.DQN(sess, major_scope_name, num_act, grid_size, num_state_frames)
	target = DQN.DQN(sess, target_scope_name, num_act, grid_size, num_state_frames)

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
		print("epoch : {}, step : {}, score : {}, avg_score : {}".format(cur_epoch, step, cur_epoch_reward, np.average([scores])))

if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("game")
	args = parser.parse_args()
	main(args.game)