# SC-DDQN
# Requirement
pycharm 2021.1.1 + Python3.6 + tf_gpu 2.1.0 + Cudnn 7.6.5.32

# Setup 
1. Install pycharm 2021.1.1 + Python3.6 + tf_gpu 2.1.0 + Cudnn 7.6.5.32
2. Configuring the environment of tf_gpu 2.1.0 and Cudnn 7.6.5.32
3. Put the point cloud data to be trained into this folder
4. Import training data, set training parameters
5. Run the program and train the Agent
6. Test the training results using the output program

# Modules Description 
1. DQN_learning.py: Main program for training Agent
2. env_qoe.py: Agent's qoe training environment
3. Exp_pool.py: Experience Pool for storing Agent training traces
4. Config.py: Relevant parameters for DRL training, including QoE-related metrics
5. model_output: For model loop output results
6. test_set_make.py: For creating data sets for simple testing

# Algorithm Principle
We design the network structure of SC-DDQN. FC stands for fully connected network and all FC layers have 761 neurons. The input and output layers, and the hidden layer all use linear rectification units. The state transmission process of this solution is also illustrated. The action $a_{1}$ is obtained at step 1, and then the next state is computed, using the new state for step 2, and so on until the whole decision variable $x_{k,c,l}$ is obtained after $m=N\times C$ steps. We use the randomly generated 100,000 sets of random parameters ${Y}$ to form the training set. In training, the network reads the training set and then starts training for up to 200,000 episodes, and the training set is trained in 200,000 episodes in a loop. The capacity of experience pool $e$ is 200,000, batch size = 256, learning rate = $1.5\times 10^{-3}$ , discount factor $\gamma = 1$. 

The state in the deep reinforcement learning algorithm includes all the variable parameters in the optimization problem P2, as well as the currently selected block identification. The whole structure is an MDP structure. The action is set to the quality level of one of the unselected slices. The reward is set as the QoE increment before and after the action is executed, and we base a large negative reward on actions that do not satisfy the constraint as a penalty.

# Statement
If you have used this code, please refer to it in your research：
J. Li et al., "Towards Optimal Real-time Volumetric Video Streaming: A Rolling Optimization and Deep Reinforcement Learning Based Approach," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2023.3277893.
