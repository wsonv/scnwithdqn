# SCN with DQN
Structured Control Nets for Deep Reinforcement Learning

Tried to apply concepts of SCN which is introduced in https://arxiv.org/abs/1802.08311 with Deep Q Network.

Hyper Parameters are randomly chosen. Thus the learning might take a while.
Because of lack of equipment, only tested whether the code is working.

The code is tested on Atari Alien-v0 and Breakout-v0.

Used environments provided by gym

Assumed all inputs are images but not RAM

Since the purpose of it is to apply SCN, for the simplicity, assumed the life per every round of game to be one.
It makes code not optimized, but enough to be trained and tested since agent must perform as good as possible for one life.


Dependencies:

Tensorflow, Gym, Numpy


References :

https://sites.google.com/view/deep-rl-bootcamp/lectures

https://medium.com/@mariosrouji/structured-control-nets-for-deep-reinforcement-learning-tutorial-icml-published-long-talk-paper-2ff99a73c8b

http://passi0n.tistory.com/88

https://tomaxent.com/2017/07/09/Using-Tensorflow-and-Deep-Q-Network-Double-DQN-to-Play-Breakout/
