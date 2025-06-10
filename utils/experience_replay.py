import numpy as np
import random

"""
define the replay buffer and corresponding algorithms like PER

"""
import random
import numpy as np
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, memory_size):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0

    def add(self, data):
        if len(self.storage) < self.memory_size:
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        idxes = random.choices(range(len(self.storage)), k=batch_size)
        samples = [self.storage[i] for i in idxes]
        batch = {key: np.array([sample[key] for sample in samples]) for key in samples[0]}
        return batch


# class replay_buffer:
#     def __init__(self, memory_size):
#         self.storge = []
#         self.memory_size = memory_size
#         self.next_idx = 0
#
#     # add the samples
#     def add(self, global_obs, private_obs, gov_action, hou_action, rewards, next_global_obs, next_private_obs, done, mean_action=None):
#         gov_reward, house_reward, firm_reward, bank_reward = rewards
#         data = (global_obs, private_obs, gov_action, hou_action, gov_reward, house_reward, firm_reward, bank_reward, next_global_obs, next_private_obs, done, mean_action)
#         if self.next_idx >= len(self.storge):
#             self.storge.append(data)
#         else:
#             self.storge[self.next_idx] = data
#         # get the next idx
#         self.next_idx = (self.next_idx + 1) % self.memory_size
#
#     # encode samples
#     def _encode_sample(self, idx):
#         global_obses, private_obses, gov_actions, hou_actions, gov_rewards, house_rewards, next_global_obses, next_private_obses, dones, mean_actions = [], [], [], [], [], [], [], [], [], []
#         for i in idx:
#             data = self.storge[i]
#             global_obs, private_obs, gov_action, hou_action,  gov_reward, house_reward, next_global_obs, next_private_obs, done, mean_action = data
#             global_obses.append(np.array(global_obs, copy=False))
#             private_obses.append(np.array(private_obs, copy=False))
#             gov_actions.append(np.array(gov_action, copy=False))
#             hou_actions.append(np.array(hou_action, copy=False))
#             gov_rewards.append(gov_reward)
#             house_rewards.append(house_reward)
#             next_global_obses.append(np.array(next_global_obs, copy=False))
#             next_private_obses.append(np.array(next_private_obs, copy=False))
#             dones.append(done)
#             mean_actions.append(mean_action)
#         return np.array(global_obses), np.array(private_obses), np.array(gov_actions), np.array(hou_actions), np.array(gov_rewards), np.array(house_rewards),\
#                np.array(next_global_obses), np.array(next_private_obses), np.array(dones), np.array(mean_actions)
#
#     # sample from the memory
#     def sample(self, batch_size):
#         idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
#         return self._encode_sample(idxes)
