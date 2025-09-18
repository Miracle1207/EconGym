import copy
import random

import numpy as np
import torch
import os, sys
# import wandb
import swanlab as wandb
import json

from omegaconf import OmegaConf

from env import EconomicSociety

sys.path.append(os.path.abspath('../..'))
from agents.log_path import make_logpath, save_args
from utils.experience_replay import ReplayBuffer
from datetime import datetime

torch.autograd.set_detect_anomaly(True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Runner:
    def __init__(self, envs, args, house_agent, government_agent, firm_agent, bank_agent):
        self.envs = copy.deepcopy(envs)
        self.args = args
        self.eval_env = copy.deepcopy(envs)

        self.house_agent = house_agent
        self.government_agent = government_agent
        self.firm_agent = firm_agent
        self.bank_agent = bank_agent
        self.agents_policy = {
            "government": self.government_agent,
            "households": self.house_agent,
            "market": self.firm_agent,
            "bank": self.bank_agent,
        }

        self.households_n = self.envs.households.households_n

        self.eva_year_indicator = 0
        self.eva_reward_indicator = 0

        # define the replay buffer
        self.buffer = ReplayBuffer(self.args.batch_size)
        self.device = 'cuda' if getattr(self.args, "cuda", False) else 'cpu'

        self.model_path, self.file_name = make_logpath(args=self.args, n=self.households_n,
                                                       task=self.envs.problem_scene)
        save_args(path=self.model_path, args=self.args)
        self.wandb = self.args.wandb

        if self.wandb:
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(self.args, resolve=True)

            wandb.init(
                config=config_dict,
                project="EconGym",
                # entity="your_account",  # TODO: Replace with your W&B account or team name
                entity="ai_tax",
                name=self.file_name + "_seed=" + str(self.args.seed),
                dir=str(self.model_path),
                job_type="training",
                # mode="offline"
            )

    def _get_tensor_inputs(self, obs_dict):
        def to_tensor(x):
            if isinstance(x, dict):
                return {k: to_tensor(v) for k, v in x.items()}
            else:
                return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        return {k: to_tensor(v) for k, v in obs_dict.items()}

    def agents_get_action(self, obs_dict_tensor):
        def act(policy, obs, path=""):
            if isinstance(policy, dict):
                return {k: act(policy[k], obs[k], f"{path}.{k}" if path else k) for k in policy}
            try:
                return policy.get_action(obs)
            except KeyError as e:
                print(f"[Warning] obs missing key at '{path}': {e}")
                return None
            except Exception as e:
                print(
                    f"[Warning] get_action failed at '{path}' for policy {getattr(policy, 'name', type(policy).__name__)}: {e}")
                return None

        return act(self.agents_policy, obs_dict_tensor)

    def run(self):
        obs_dict = self.envs.reset()

        for epoch in range(self.args.n_epochs):
            transition_dict = {
                "obs_dict": [],
                "action_dict": [],
                "reward_dict": [],
                "next_obs_dict": [],
                "done": []
            }

            sum_loss = {
                "actor_loss": {},
                "critic_loss": {}
            }

            for t in range(self.args.epoch_length):
                obs_dict_tensor = self._get_tensor_inputs(obs_dict)
                action_dict = self.agents_get_action(obs_dict_tensor)
                next_obs_dict, reward_dict, done = self.envs.step(action_dict, t)

                on_policy_process = all(self.envs.recursive_decompose_dict(self.agents_policy, lambda a: a.on_policy))

                if on_policy_process:
                    # on policy
                    for key in transition_dict:
                        transition_dict[key].append(locals()[key])
                else:
                    # off-policy
                    for key in transition_dict:
                        transition_dict[key] = (locals()[key])
                    self.buffer.add(transition_dict)

                obs_dict = next_obs_dict
                if done:
                    obs_dict = self.envs.reset()

            for agent_name in self.agents_policy:
                sub_agent_policy = self.agents_policy[agent_name]
                batch_size = self.args.epoch_length if on_policy_process else self.args.batch_size
                agent_data = self.buffer.sample(agent_name=agent_name, agent_policy=sub_agent_policy,
                                                batch_size=batch_size, on_policy=on_policy_process,
                                                transition_dict=transition_dict)
                if isinstance(sub_agent_policy, dict):
                    for name in sub_agent_policy:
                        sum_loss = self.sub_agent_training(agent_name=name,
                                                           agent_policy=sub_agent_policy[name],
                                                           transitions=agent_data[name],
                                                           loss=sum_loss)
                else:
                    sum_loss = self.sub_agent_training(agent_name=agent_name,
                                                       agent_policy=sub_agent_policy,
                                                       transitions=agent_data,
                                                       loss=sum_loss)

            # print the log information
            if epoch % self.args.display_interval == 0:
                economic_idicators_dict = self._evaluate_agent()

                if self.wandb:
                    wandb.log(economic_idicators_dict)
                    wandb.log(sum_loss)

                print(
                    '[{}] Epoch: {} / {}, Frames: {}, Gov_Rewards: {:.3f}, House_Rewards: {:.3f}, Firm_Rewards: {:.3f}, Bank_Rewards: {:.3f},  years:{:.3f}'.format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length,
                        economic_idicators_dict["gov_reward"], economic_idicators_dict["house_reward"],
                        economic_idicators_dict["firm_reward"], economic_idicators_dict["bank_reward"],
                        economic_idicators_dict["years"]))

            if epoch % self.args.save_interval == 0:  # save_interval=10
                self.envs.recursive_decompose_dict(self.agents_policy, lambda a: a.save(dir_path=self.model_path))

        if self.wandb:
            wandb.finish()

    def sub_agent_training(self, agent_name, agent_policy, transitions, loss):
        if agent_policy.on_policy == True:
            actor_loss, critic_loss = agent_policy.train(transitions)
            loss['actor_loss'][agent_name] = actor_loss
            loss['critic_loss'][agent_name] = critic_loss
        else:
            total_actor_loss = 0.
            total_critic_loss = 0.
            for _ in range(self.args.update_cycles):
                actor_loss, critic_loss = agent_policy.train(transitions)
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
            loss['actor_loss'][agent_name] = total_actor_loss
            loss['critic_loss'][agent_name] = total_critic_loss
        return loss

    def test(self):
        ''' record the actions of gov and households'''
        economic_idicators_dict = self._evaluate_agent(write_evaluate_data=False)

    def viz_data(self, house_model_path, government_model_path):
        self.house_agent.load(dir_path=house_model_path)
        self.government_agent.load(dir_path=government_model_path)
        # this data is used for visualization
        self._evaluate_agent(write_evaluate_data=True)

    def init_economic_dict(self, reward_dict):
        gov_rewards = reward_dict['government']
        households_reward = reward_dict['households']
        firm_reward = reward_dict['market']
        bank_reward = reward_dict['bank']

        gov_reward = sum([reward_dict['government'][key] for key in reward_dict['government']])

        self.econ_dict = {
            "gov_reward": gov_reward,  # sum
            "tax_gov_reward": gov_rewards.get('tax', 0),
            "central_bank_gov_reward": gov_rewards.get('central_bank', 0),
            "pension_gov_reward": gov_rewards.get('pension', 0),
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
            "house_consumption": self.eval_env.households.consumption,
            "house_work_hours": self.eval_env.households.ht,
            "gov_spending": self.eval_env.main_gov.gov_spending,
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
        eval_econ = ["gov_reward", "tax_gov_reward", "central_bank_gov_reward", "pension_gov_reward",
                     "house_reward", "social_welfare", "per_gdp", "income_gini",
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
        obs_dict = self.eval_env.reset()
        episode_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
        final_econ_dict = dict(zip(eval_econ, [None for i in range(len(eval_econ))]))

        for epoch_i in range(self.args.eval_episodes):
            eval_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
            t = 0
            while True:
                with torch.no_grad():
                    obs_dict_tensor = self._get_tensor_inputs(obs_dict)
                    action_dict = self.agents_get_action(obs_dict_tensor)
                    next_obs_dict, rewards_dict, done = self.eval_env.step(action_dict, t)
                t += 1
                self.init_economic_dict(rewards_dict)

                for each in eval_econ:
                    if "house_" in each or each == "WageRate":
                        eval_econ_dict[each].append(self.econ_dict[each].tolist())
                    else:
                        eval_econ_dict[each].append(self.econ_dict[each])

                obs_dict = next_obs_dict
                if done:
                    obs_dict = self.eval_env.reset()
                    break

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
            file_name = f"{self.file_name}_data.json"
            # f"{self.eval_env.problem_scene}_{self.eval_env.households.type}_{self.households_n}_{self.args.house_alg}_" \
            #         f"{self.eval_env.main_gov.type}_{self.gov_alg}_data.json"

            file_path = os.path.join(store_path, file_name)
            with open(file_path, "w") as file:
                json.dump(eval_econ_dict, file, cls=NumpyEncoder)

            print("============= Finish Writing================")

        return final_econ_dict
