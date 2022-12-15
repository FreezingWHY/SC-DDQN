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
