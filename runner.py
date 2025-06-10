import copy
import random

import numpy as np
import torch
import os, sys
import wandb
import json

from omegaconf import OmegaConf

from env import EconomicSociety

sys.path.append(os.path.abspath('../..'))
from agents.log_path import make_logpath
from utils.experience_replay import ReplayBuffer
from datetime import datetime

torch.autograd.set_detect_anomaly(True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Runner:
    def __init__(self, envs, args, house_agent, firm_agent, bank_agent, central_bank_gov_agent=None, tax_gov_agent=None,
                 pension_gov_agent=None, heter_house=None, government_agent=None):
        self.envs = copy.deepcopy(envs)
        self.args = args
        self.eval_env = copy.deepcopy(envs)
        self.gov_agents = []

        self.house_agent = house_agent
        if government_agent:
            self.government_agent = government_agent
            self.gov_agents.append(government_agent)
            self.main_gov_agent = government_agent
        if central_bank_gov_agent:
            self.central_bank_gov_agent = central_bank_gov_agent
            self.gov_agents.append(central_bank_gov_agent)
        if tax_gov_agent:
            self.tax_gov_agent = tax_gov_agent
            self.gov_agents.append(tax_gov_agent)
            self.main_gov_agent = tax_gov_agent
        if pension_gov_agent:
            self.pension_gov_agent = pension_gov_agent
            self.gov_agents.append(pension_gov_agent)

        self.firm_agent = firm_agent
        self.bank_agent = bank_agent
        self.households_n = self.envs.households.households_n
        
        if self.args.heterogeneous_house_agent == True:
            self.heter_house = heter_house
        # define the replay buffer
        self.buffer = ReplayBuffer(self.args.buffer_size)


        gov_alg_mapping = {
            'central_bank_gov': 'central_bank_gov_alg',
            'pension_gov': 'pension_gov_alg',
            'tax_gov': 'tax_gov_alg',
            'government': 'gov_alg',
        }

        gov_alg_key = gov_alg_mapping.get(envs.main_gov.name, 'gov_alg')  #
        self.gov_alg = getattr(args, gov_alg_key, 'rule_based')  #

        self.model_path, _ = make_logpath(algo=self.args.house_alg + "_" + self.gov_alg,
                                          n=self.households_n, task=self.envs.main_gov.gov_task)

        save_args(path=self.model_path, args=self.args)
        self.households_welfare = 0
        self.eva_year_indicator = 0
        self.eva_reward_indicator = 0
        self.wandb = self.args.wandb
        self.aligned = self.args.aligned
        np.random.seed(self.args.seed)
        
        if self.wandb:
            wandb.init(
                config=self.args,
                project="MACRO",
                entity="ai_tax",
                name=self.model_path.parent.parent.parent.name + "_" + self.model_path.name + '_' + str(
                    self.households_n) + "_" + self.envs.gov_task + "_seed=" + str(self.args.seed),
                dir=str(self.model_path),
                job_type="training",
                reinit=True
            )
    
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        return obs_tensor
    
    def get_obs(self, current_env, agent_name, global_obs_tensor, private_obs_tensor, gov_action, bank_action,
                firm_action=None):
        obs_dim = current_env.agents[agent_name].observation_space.shape[0]
        
        # Generate previous_action tensor
        gov_bank_action = np.concatenate([gov_action, bank_action])
        if agent_name == current_env.households.name:
            if obs_dim == gov_action.size + bank_action.size + firm_action.size + private_obs_tensor.shape[-1]:
                # Flatten firm_action for consistency
                firm_action = firm_action.flatten()
                previous_action = self._get_tensor_inputs(np.concatenate([gov_bank_action, firm_action]))
                previous_action_repeated = previous_action.repeat(current_env.households.households_n, 1)
                return torch.cat((private_obs_tensor, previous_action_repeated), dim=-1)
            elif obs_dim == private_obs_tensor.shape[-1]:
                return private_obs_tensor
        elif agent_name == current_env.market.name:
            if obs_dim == gov_action.size + bank_action.size + global_obs_tensor.shape[-1] + 1:
                previous_action = self._get_tensor_inputs(gov_bank_action)
                base_obs = torch.cat((global_obs_tensor, previous_action), dim=-1)  # shape: [1, D]
                firm_n = current_env.market.Zt.shape[0]
                # Expand to (firm_n, D)
                expanded_obs = base_obs.expand(firm_n, -1)
                # Concatenate Zt: shape (firm_n, D+1)
                full_obs = torch.cat((expanded_obs, self._get_tensor_inputs(current_env.market.Zt)), dim=-1)
        
                return full_obs
    
            elif obs_dim == global_obs_tensor.shape[-1]:
                return None
        
        # Raise error if no matching condition is found
        raise ValueError(f"Unexpected {agent_name} observation dimension: {obs_dim}")
    
    def agents_get_action(self, env, global_obs_tensor, private_obs_tensor):

        gov_action = self.main_gov_agent.get_action(global_obs_tensor, private_obs_tensor, env=env)
        bank_action = self.bank_agent.get_action(global_obs_tensor, private_obs_tensor)
        firm_private_obs_tensor = self.get_obs(env, 'market', global_obs_tensor, private_obs_tensor, gov_action, bank_action)
        if hasattr(self, 'central_bank_gov_agent'):
            central_bank_gov_action = self.central_bank_gov_agent.get_action(global_obs_tensor, private_obs_tensor,
                                                                             env=env)
        if hasattr(self, 'pension_gov_agent'):
            pension_gov_action = self.pension_gov_agent.get_action(global_obs_tensor, private_obs_tensor, env=env)

        firm_action = self.firm_agent.get_action(global_obs_tensor, firm_private_obs_tensor)
        house_private_obs_tensor = self.get_obs(env, 'household', global_obs_tensor, private_obs_tensor,
                                                gov_action, bank_action, firm_action)

        house_action = self.house_agent.get_action(global_obs_tensor, house_private_obs_tensor, env=env)

        # if use mean field method
        if "mf" in self.args.house_alg:
            house_action, mean_house_action = house_action
        else:
            mean_house_action = None
    

        action_dict = {self.envs.main_gov.name: gov_action,
                       self.envs.bank.name: bank_action,
                       self.envs.market.name: firm_action,
                       self.envs.households.name: house_action}
        private_obs_dict = {self.envs.main_gov.name: private_obs_tensor,
                            self.envs.bank.name: private_obs_tensor,
                            self.envs.market.name: firm_private_obs_tensor,
                            self.envs.households.name: house_private_obs_tensor}
        if hasattr(self, 'pension_gov_agent'):
            action_dict[self.envs.pension_gov.name] = pension_gov_action
            private_obs_dict[self.envs.pension_gov.name] = private_obs_tensor

        if hasattr(self, 'central_bank_gov_agent'):
            action_dict[self.envs.central_bank_gov.name] = central_bank_gov_action
            private_obs_dict[self.envs.central_bank_gov.name] = private_obs_tensor

        return action_dict, private_obs_dict, mean_house_action
    
    def run(self):
        agents = [self.house_agent] + self.gov_agents

        gov_rew, house_rew, epochs = [], [], []
        global_obs, private_obs = self.envs.reset()
        
        for epoch in range(self.args.n_epochs):
            transition_dict = {'global_obs': [], 'private_obs': [], 'gov_action': [], 'house_action': [],
                               'gov_reward': [], 'house_reward': [], 'firm_reward': [], 'bank_reward': [],
                               'next_global_obs': [], 'next_private_obs': [], 'done': [], "mean_house_actions": []}

            if hasattr(self, 'pension_gov_agent'):
                transition_dict['pension_gov_action'] = []
                transition_dict['pension_gov_reward'] = []

            if hasattr(self, 'central_bank_gov_agent'):
                transition_dict['central_bank_gov_action'] = []
                transition_dict['central_bank_gov_reward'] = []

            sum_loss = np.zeros((len(agents), 2))
            for t in range(self.args.epoch_length):
                global_obs_tensor = self._get_tensor_inputs(global_obs)
                private_obs_tensor = self._get_tensor_inputs(private_obs)
                action_dict, private_obs_dict, mean_house_action = self.agents_get_action(self.envs, global_obs_tensor,
                                                                                          private_obs_tensor)
                # 给household
                next_global_obs, next_private_obs, gov_reward, house_reward, firm_reward, bank_reward, done = self.envs.step(
                    action_dict, t)

                on_policy_process = any(agent.on_policy for agent in agents)
                if on_policy_process:
                    # on policy
                    transition_dict['global_obs'].append(global_obs)
                    transition_dict['private_obs'].append(private_obs)
                    transition_dict['gov_action'].append(action_dict[self.envs.main_gov.name])

                    if hasattr(self, 'pension_gov_agent'):
                        transition_dict['pension_gov_action'].append(action_dict[self.envs.pension_gov.name])
                    if hasattr(self, 'central_bank_gov_agent'):
                        transition_dict['central_bank_gov_action'].append(action_dict[self.envs.central_bank_gov.name])

                    transition_dict['house_action'].append(action_dict[self.envs.households.name])
                    transition_dict['gov_reward'].append(gov_reward)
                    transition_dict['house_reward'].append(house_reward)
                    transition_dict['firm_reward'].append(firm_reward)
                    transition_dict['bank_reward'].append(bank_reward)
                    transition_dict['next_global_obs'].append(next_global_obs)
                    transition_dict['next_private_obs'].append(next_private_obs)
                    transition_dict['done'].append(float(done))
                    transition_dict['mean_house_actions'].append(mean_house_action)

                off_policy_process = any(not agent.on_policy for agent in agents)
                if off_policy_process:
                    # off policy: replay buffer
                    data = {
                        'global_obs': global_obs,
                        # 'private_obs': private_obs,
                        'gov_action': action_dict[self.envs.main_gov.name],
                        # 'hou_action': action_dict[self.envs.households.name],
                        'gov_reward': gov_reward,
                        # 'house_reward': house_reward,
                        'next_global_obs': next_global_obs,
                        # 'next_private_obs': next_private_obs,
                        'done': done,
                        # 'mean_action': mean_house_action
                    }
                    
                    if hasattr(self, 'pension_gov_agent'):
                        data['pension_gov_action'] = action_dict[self.envs.pension_gov.name]
                    if hasattr(self, 'central_bank_gov_agent'):
                        data['central_bank_gov_action'] = action_dict[self.envs.central_bank_gov.name]

                    self.buffer.add(data)
                
                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    global_obs, private_obs = self.envs.reset()
            
            for i in range(len(agents)):
                # if epoch < 10 or epoch % 5 == 0 or i == 1:
                if agents[i].on_policy == True:
                    actor_loss, critic_loss = agents[i].train(transition_dict)
                    sum_loss[i, 0] = actor_loss
                    sum_loss[i, 1] = critic_loss
                else:
                    for _ in range(self.args.update_cycles):
                        transitions = self.buffer.sample(self.args.batch_size)
                        actor_loss, critic_loss = agents[i].train(transitions,
                                                                  other_agent=agents[1 - i])  # MARL has other agents
                        sum_loss[i, 0] += actor_loss
                        sum_loss[i, 1] += critic_loss
            
            # print the log information
            if epoch % self.args.display_interval == 0:  # display_interval=1
                economic_idicators_dict = self._evaluate_agent()
                now_step = (epoch + 1) * self.args.epoch_length
                gov_rew.append(economic_idicators_dict["gov_reward"])
                house_rew.append(economic_idicators_dict["house_reward"])
                np.savetxt(str(self.model_path) + "/gov_reward.txt", gov_rew)
                np.savetxt(str(self.model_path) + "/house_reward.txt", house_rew)
                epochs.append(now_step)
                np.savetxt(str(self.model_path) + "/steps.txt", epochs)
                loss_dict = {
                    "house_actor_loss": sum_loss[1, 0],
                    "house_critic_loss": sum_loss[1, 1],
                    "gov_actor_loss": sum_loss[0, 0],
                    "gov_critic_loss": sum_loss[0, 1]
                }
                
                if self.wandb:
                    wandb.log(economic_idicators_dict)
                    wandb.log(loss_dict)
                
                print(
                    '[{}] Epoch: {} / {}, Frames: {}, Gov_Rewards: {:.3f}, House_Rewards: {:.3f}, Firm_Rewards: {:.3f}, Bank_Rewards: {:.3f},  years:{:.3f}, actor_loss: {:.3f}, critic_loss: {:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length,
                        economic_idicators_dict["gov_reward"], economic_idicators_dict["house_reward"],
                        economic_idicators_dict["firm_reward"], economic_idicators_dict["bank_reward"],
                        economic_idicators_dict["years"], np.sum(sum_loss[:, 0]), np.sum(sum_loss[:, 1])))
            
            if epoch % self.args.save_interval == 0:  # save_interval=10
                self.house_agent.save(dir_path=self.model_path)
                self.main_gov_agent.save(dir_path=self.model_path)

        if self.wandb:
            wandb.finish()
    
    def test(self):
        ''' record the actions of gov and households'''
        economic_idicators_dict = self._evaluate_agent(write_evaluate_data=False)
    
    def viz_data(self, house_model_path, government_model_path):
        self.house_agent.load(dir_path=house_model_path)
        self.government_agent.load(dir_path=government_model_path)
        # this data is used for visualization
        self._evaluate_agent(write_evaluate_data=True)

    def init_economic_dict(self, gov_reward, households_reward, firm_reward, bank_reward):
        self.econ_dict = {
            "gov_reward": gov_reward,  # sum
            "social_welfare": np.sum(households_reward),  # sum
            "house_reward": households_reward,  # sum
            "firm_reward": firm_reward,
            "bank_reward": bank_reward,
            "years": self.eval_env.step_cnt,  # max
            "house_income": self.eval_env.households.post_income,  # post_tax income
            "house_total_tax": self.eval_env.main_gov.tax_array,
            "house_income_tax": self.eval_env.households.income_tax,
            "house_pension": self.eval_env.households.pension,
            "house_wealth": self.eval_env.households.at_next,
            "house_wealth_tax": self.eval_env.households.asset_tax,
            "per_gdp": self.eval_env.main_gov.per_household_gdp,
            "GDP": self.eval_env.main_gov.GDP,  # sum
            "income_gini": self.eval_env.income_gini,
            "wealth_gini": self.eval_env.wealth_gini,
            "WageRate": self.eval_env.market.WageRate,
            "total_labor": self.eval_env.market.Lt,
            "house_consumption": self.eval_env.households.consumption_ij,
            "house_work_hours": self.eval_env.households.ht,
            "gov_spending": self.eval_env.main_gov.Gt_prob * self.eval_env.main_gov.GDP,
            "house_age": self.eval_env.households.age,
        }
        if hasattr(self, 'pension_gov_agent'):
            self.econ_dict['retire_age'] = self.eval_env.pension_gov.retire_age
            self.econ_dict['contribution_rate'] = self.eval_env.pension_gov.contribution_rate
            self.econ_dict['pension_fund'] = self.eval_env.pension_gov.pension_fund
            self.econ_dict['old_percent'] = self.eval_env.pension_gov.old_percent
            self.econ_dict['dependency_ratio'] = self.eval_env.pension_gov.dependency_ratio

    def sum_non_uniform_dict(self, sequences):
        total_sum = 0
        for sublist in sequences:
            if isinstance(sublist, list) or isinstance(sublist, np.ndarray):
                sublist_sum = np.sum(sublist)
                total_sum += sublist_sum
            else:
                raise ValueError("Unsupported data type within the sequence")
        return total_sum
    
    def mean_non_uniform_dict(self, sequences):
        flat_list = [item for sublist in sequences for item in
                     (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
        return np.mean(flat_list)

    def _evaluate_agent(self, write_evaluate_data=False):
        eval_econ = ["gov_reward", "house_reward", "social_welfare", "per_gdp", "income_gini",
                     "wealth_gini", "years", "GDP", "gov_spending", "house_total_tax", "house_income_tax",
                     "house_wealth_tax", "house_wealth", "house_income", "house_consumption", "house_pension",
                     "house_work_hours", "total_labor", "WageRate", "house_age", "firm_reward", "bank_reward"]

        if hasattr(self, 'pension_gov_agent'):
            eval_econ += [
                "retire_age",
                "contribution_rate",
                "pension_fund",
                "old_percent",
                "dependency_ratio"
            ]
        global_obs, private_obs = self.eval_env.reset()
        episode_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
        final_econ_dict = dict(zip(eval_econ, [None for i in range(len(eval_econ))]))
        for epoch_i in range(self.args.eval_episodes):
            eval_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
            
            while True:
                with torch.no_grad():
                    global_obs_tensor = self._get_tensor_inputs(global_obs)
                    private_obs_tensor = self._get_tensor_inputs(private_obs)
                    action_dict, private_obs_dict, mean_house_action = self.agents_get_action(self.eval_env,
                                                                                              global_obs_tensor,
                                                                                              private_obs_tensor)
                    
                    next_global_obs, next_private_obs, gov_reward, house_reward, firm_reward, bank_reward, done = self.eval_env.step(
                        action_dict)

                self.init_economic_dict(gov_reward, house_reward, firm_reward, bank_reward)

                for each in eval_econ:
                    if "house_" in each or each == "WageRate":
                        eval_econ_dict[each].append(self.econ_dict[each].tolist())
                    else:
                        eval_econ_dict[each].append(self.econ_dict[each])
                
                global_obs = next_global_obs
                private_obs = next_private_obs
                if done:
                    global_obs, private_obs = self.eval_env.reset()
                    break
            # print(f"epoch {epoch_i}  mean_a {np.mean(self.eval_env.market.Kt_next)}")
            
            
            for key, value in eval_econ_dict.items():
                if key == "gov_reward" or key == "GDP" or key == "bank_reward" or key == "firm_reward":  # You can store the firm_reward for each firm individually.
                    episode_econ_dict[key].append(np.sum(value))
                elif key == "years":
                    episode_econ_dict[key].append(np.max(value))
                elif key == "house_reward":
                    episode_econ_dict[key].append(self.sum_non_uniform_dict(value))
                elif "house_" in key and key != "house_reward":
                    episode_econ_dict[key].append(self.mean_non_uniform_dict(value))
                elif key == "WageRate":
                    WageRate_flattened = np.array([item[0][0] for item in value])
                    episode_econ_dict[key].append(np.mean(WageRate_flattened))
                elif key == "age":
                    episode_econ_dict[key].append(value)
                else:
                    episode_econ_dict[key].append(np.mean(value))

        for key, value in episode_econ_dict.items():
            final_econ_dict[key] = np.mean(value)

        if int(self.econ_dict['years']) > int(self.eva_year_indicator):
            write_evaluate_data = True
            self.eva_year_indicator = self.econ_dict['years']
        elif self.econ_dict['years'] == self.eva_year_indicator:
            gov_return = np.sum(eval_econ_dict['gov_reward'])
            if gov_return > self.eva_reward_indicator:
                write_evaluate_data = True
                self.eva_reward_indicator = copy.deepcopy(gov_return)
        
        # write_evaluate_data=False
        if write_evaluate_data:
            store_path = "viz/data/"
            if not os.path.exists(store_path):  # 确保路径存在
                os.makedirs(store_path)
            file_name = f"{self.eval_env.problem_scene}_{self.eval_env.households.type}_{self.households_n}_{self.args.house_alg}_" \
                        f"{self.eval_env.main_gov.type}_{self.gov_alg}_data.json"

            file_path = os.path.join(store_path, file_name)
            with open(file_path, "w") as file:
                json.dump(eval_econ_dict, file, cls=NumpyEncoder)

            print("============= Finish Writing================")

        return final_econ_dict


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

