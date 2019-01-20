import tensorflow as tf

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

