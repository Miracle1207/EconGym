import copy
import numpy as np
import torch
import os, sys
import pandas as pd
import wandb
import time

from scipy.stats import truncnorm

sys.path.append(os.path.abspath('..'))

torch.autograd.set_detect_anomaly(True)


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class data_agent:
    def __init__(self, envs, args, agent_name=None, type=None):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name

        # get the action max
        # self.gov_action_max = self.envs.government.action_space.high[0]
        # self.hou_action_max = self.envs.households.action_space.high[0]
        self.on_policy = True

    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, bank_action=None,
                   firm_action=None, env=None):
        if self.agent_name == "government":
            gov_action = np.zeros(self.envs.government.action_dim)
            if self.envs.government.type == "tax":
                gov_action = np.concatenate((np.array([0.263, 0.049, 0., 0., 0.189]),
                                             np.ones(self.envs.government.action_dim - 5) / (
                                                     self.envs.government.action_dim - 5)))

            elif self.envs.government.type == "central_bank":
                gov_action = np.concatenate(
                    (np.array([-0.02, 0.08]), np.random.randn(self.envs.government.action_dim - 2)))
            # pension
            elif self.envs.government.type == "pension":  # cite IMF (2020); OECD (2019) 动态响应机制

                gov_action = np.concatenate((np.array([67, 0.08]), np.ones(self.envs.government.action_dim - 2) / (
                        self.envs.government.action_dim - 2)))  # 固定数字
            else:
                print("Wrong government agent type!")
            return gov_action
        elif self.agent_name == "households":
            house_action = np.random.randn(self.envs.households.households_n, self.envs.households.action_dim)

            house_action[:, 1] = np.clip(house_action[:, 1], 0, 1)  # Ensure in [0, 1]

            return house_action

        elif self.agent_name == "market":
            firm_action = np.random.randn(self.envs.market.firm_n, self.envs.market.action_dim)
            return firm_action
        elif self.agent_name == "bank":
            bank_action = np.random.randn(self.envs.bank.action_dim)
            return bank_action

        elif self.agent_name == "central_bank_gov":
            central_bank_gov_action = np.concatenate(
                (np.array([-0.02, 0.08]), np.random.randn(self.envs.central_bank_gov.action_dim - 2)))
            return central_bank_gov_action

        elif self.agent_name == "pension_gov":
            pension_gov_action = np.concatenate((np.array([67, 0.08]), np.ones(self.envs.pension_gov.action_dim - 2) / (
                    self.envs.pension_gov.action_dim - 2)))  # 固定数字
            return pension_gov_action

        elif self.agent_name == "tax_gov":
            tax_gov_action = np.concatenate((np.array([0.263, 0.049, 0., 0., 0.189]),
                                             np.ones(self.envs.tax_gov.action_dim - 5) / (
                                                     self.envs.tax_gov.action_dim - 5)))
            return tax_gov_action

    def train(self, transition):
        return 0, 0

    def save(self, dir_path, step=0):
        pass
