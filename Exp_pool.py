import numpy as np
import random
class Exp_pool:
    def __init__(self,config):
        self.memory_size = config.memory_size
        self.states = np.empty((self.memory_size,config.s_len))
        self.actions = np.empty(self.memory_size,dtype = np.integer)
        self.rewards = np.empty(self.memory_size)
        self.next_states = np.empty((self.memory_size,config.s_len))
        self.if_done = np.empty(self.memory_size)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0
    def add(self, state, action, reward, next_states,if_done):
        self.states[self.current] = state
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.next_states[self.current] = next_states
        self.if_done[self.current] = if_done

        self.count = max(self.count, self.current + 1)              #当前池中已占用位置数,存了几个就是几
        self.current = (self.current + 1) % self.memory_size        #当前位置标记，包含遗忘机制
        return

    def get_batch(self):
        # sample random indexes
        if(self.count<self.memory_size):
            up = self.count
        else:
            up = self.memory_size
        indexes = random.sample(range(0, up), self.batch_size)
        indexes = sorted(indexes)
        states = self.states[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        next_states = self.next_states[indexes]
        if_done = self.if_done[indexes]
        return states,actions,rewards,next_states,if_done

# if __name__ == '__main__':


#     #1
#     config = Config()
#     pool = Exp_pool(config)
#     state = [1,3,2,5,6,4,34,1,2,3,0,0,0,0,0,0,0,0,0,0]
#     action = 1
#     reward = 1
#     next_states = [1,3,2,5,6,4,34,1,2,3,0,1,0,0,1,0,0,0,0,0]
#     if_done = 3
#     pool.add(state,action,reward,next_states,if_done)
#     #2
#     state = [2, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     action = 2
#     reward = 2
#     next_states = [2, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
#     if_done = 3
#     pool.add(state,action,reward,next_states,if_done)
#     #3
#     state = [3, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     action = 3
#     reward = 3
#     next_states = [3, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
#     if_done = 3
#     pool.add(state,action,reward,next_states,if_done)
#     #4
#     state = [4, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     action = 4
#     reward = 4
#     next_states = [4, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
#     if_done = 4
#     pool.add(state,action,reward,next_states,if_done)
#     #5
#     state = [5, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     action = 5
#     reward = 5
#     next_states = [5, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
#     if_done = 5
#     pool.add(state,action,reward,next_states,if_done)
#     #6
#     state = [6, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     action = 6
#     reward = 6
#     next_states = [6, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
#     if_done = 6
#     pool.add(state,action,reward,next_states,if_done)
#     #7
#     state = [7, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     action = 7
#     reward = 7
#     next_states = [7, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
#     if_done = 7
#     pool.add(state,action,reward,next_states,if_done)
#     #8
#     state = [8, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     action = 8
#     reward = 8
#     next_states = [8, 99, 2, 5, 6, 4, 34, 1, 2, 3, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
#     if_done = 8
#     pool.add(state,action,reward,next_states,if_done)
#
#
#
#     states,actions,rewards,next_states,if_dones= pool.get_batch()
#     print(states)


    # for i in range(3) :

        # print(pool.states[i])
        # print(pool.actions[i])
        # print(pool.rewards[i])
        # print(pool.next_states[i])
        # print(pool.if_done[i])
        # print("current=",pool.current)
        # print("count=",pool.count)

    # print(pool.states)
    # print(pool.actions)
    # print(pool.rewards)
    # print(pool.next_states)
    # print(pool.if_done)
