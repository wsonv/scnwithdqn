# SCN with DQN
Structured Control Nets for Deep Reinforcement Learning

Tried to apply concepts of SCN which is introduced on https://arxiv.org/abs/1802.08311 with Deep Q Network.

Code is simple.
Commented on the code for detailed explanation.
SCN is applied when building CNN.

Hyper Parameters are randomly chosen. Thus the learning might take a while.
Because of lack of equipments, the code is tested only for whether it is working without errors.

Used environments provided by gym.

The code is tested on Atari Alien-v0, Breakout-v0 and, MsPacman-v0.


Assumed all inputs are images but not RAM.

Since the purpose of it is to apply SCN, for the simplicity, assumed the life per every round of game to be one.
It makes code not optimized, but enough to be trained and tested since agent must perform as good as possible for one life.


### **Dependencies**:

Tensorflow, Gym, Numpy



### **Python version** : 

3.6


### **Execution method** :

Clone this repository and type below on command line

python SCN_with_DQN.py "Name of the game you want to test"

ex) python SCN_with_DQN.py "Alien-v0"


### **References** :

https://sites.google.com/view/deep-rl-bootcamp/lectures

https://medium.com/@mariosrouji/structured-control-nets-for-deep-reinforcement-learning-tutorial-icml-published-long-talk-paper-2ff99a73c8b

https://gist.github.com/jcwleo/fffc40f69b7f14d9e2a2b8765a79b579

https://tomaxent.com/2017/07/09/Using-Tensorflow-and-Deep-Q-Network-Double-DQN-to-Play-Breakout/
