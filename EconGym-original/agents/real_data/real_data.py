import numpy as np
import torch
import os, sys

sys.path.append(os.path.abspath('../../..'))
from agents.rl.models import CloneModel
import copy
import pickle
from torch.distributions.normal import Normal


def load_params_from_file(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class real_agent:
    def __init__(self, envs, args, agent_name="households"):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        house_obs_dim = 2
        house_action_dim = self.envs.households.action_space.shape[1]
        # start to build the network.
        self.households_net = CloneModel(house_obs_dim, house_action_dim)
        self.households_net.load_state_dict(
            torch.load("agents/real_data/2023_11_23_20_37_trained_model.pth"))  # v1 nice!
        if self.args.cuda:
            self.households_net.cuda()
        # get the action max
        self.gov_action_max = self.envs.government.action_space.high[0]
        self.hou_action_max = self.envs.households.action_space.high[0]
        self.on_policy = True

    def get_action(self, global_obs_tensor, private_obs_tensor, households_age=None, agent_name="households"):
        if agent_name == "government":
            print("Error: this is households' policy!")
        elif agent_name == "households":
            temp = private_obs_tensor[:, 1]
            private_obs_tensor[:, 1] = private_obs_tensor[:, 0]
            private_obs_tensor[:, 0] = temp
            households_n = len(private_obs_tensor)

            temp = np.random.random((households_n, self.envs.households.action_space.shape[1]))
            mean, std = self.households_net(private_obs_tensor)
            # Create a normal distribution
            dist = Normal(mean, std)
            # Sample an action
            sampled_actions = dist.sample().cpu().numpy()
            temp[:, 0] = 1 - sampled_actions[:, 1]
            temp[:, 1] = sampled_actions[:, 0]
            hou_action = temp

            return hou_action

    def train(self, transition):
        return 0, 0

    def save(self, dir_path, step=0):
        pass
