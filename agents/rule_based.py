import copy
import numpy as np
import torch
import os, sys
import pandas as pd
import wandb
import time

from scipy.stats import truncnorm

sys.path.append(os.path.abspath('../..'))
torch.autograd.set_detect_anomaly(True)


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class rule_agent:
    def __init__(self, envs, args, agent_name=None):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        self.on_policy = True

    def generate_saving_proportion(self, age, households_n, country):
        """
        Generate saving proportion based on age (for OLG) or population average (for Ramsey).

        Parameters:
            age: numpy array of ages, or None for Ramsey

        Returns:
            saving_proportion: numpy array of saving proportions in [0, 1]
        """
        if age is not None:
            saving_proportion = np.zeros_like(age, dtype=float)
            for i, a in enumerate(age):
                if country == 'US':
                    if a < 35:
                        saving_proportion[i] = np.random.normal(0.25, 0.08)
                    elif 35 <= a < 45:
                        saving_proportion[i] = np.random.normal(0.35, 0.08)
                    elif 45 <= a < 55:
                        saving_proportion[i] = np.random.normal(0.45, 0.08)
                    elif 55 <= a < 65:
                        saving_proportion[i] = np.random.normal(0.55, 0.06)
                    else:
                        saving_proportion[i] = np.random.normal(0.65, 0.06)
                else:  # China
                    if a < 35:
                        saving_proportion[i] = np.random.normal(0.35, 0.10)
                    elif 35 <= a < 45:
                        saving_proportion[i] = np.random.normal(0.45, 0.10)
                    elif 45 <= a < 55:
                        saving_proportion[i] = np.random.normal(0.55, 0.10)
                    elif 55 <= a < 65:
                        saving_proportion[i] = np.random.normal(0.65, 0.08)
                    else:
                        saving_proportion[i] = np.random.normal(0.75, 0.08)
        else:
            # Ramsey: age-independent
            mean = 0.40 if country == 'US' else 0.50
            std = 0.08 if country == 'US' else 0.10
            saving_proportion = np.random.normal(mean, std, size=households_n)

        return np.clip(saving_proportion, 0, 1)

    def generate_risk_investment_proportion(self, age, households_n, country):
        """
        Generate risk investment proportion (fraction of savings in risky assets).

        Parameters:
            age: numpy array of ages, or None for Ramsey

        Returns:
            risk_invest_proportion: numpy array of risk investment proportions in [0, 1]
        """
        if age is not None:
            risk_invest_proportion = np.zeros_like(age, dtype=float)
            for i, a in enumerate(age):
                if country == 'US':
                    if a < 35:
                        risk_invest_proportion[i] = np.random.normal(0.70, 0.10)
                    elif 35 <= a < 45:
                        risk_invest_proportion[i] = np.random.normal(0.60, 0.10)
                    elif 45 <= a < 55:
                        risk_invest_proportion[i] = np.random.normal(0.50, 0.10)
                    elif 55 <= a < 65:
                        risk_invest_proportion[i] = np.random.normal(0.35, 0.08)
                    else:
                        risk_invest_proportion[i] = np.random.normal(0.25, 0.08)
                else:  # China
                    if a < 35:
                        risk_invest_proportion[i] = np.random.normal(0.50, 0.12)
                    elif 35 <= a < 45:
                        risk_invest_proportion[i] = np.random.normal(0.40, 0.12)
                    elif 45 <= a < 55:
                        risk_invest_proportion[i] = np.random.normal(0.35, 0.10)
                    elif 55 <= a < 65:
                        risk_invest_proportion[i] = np.random.normal(0.25, 0.08)
                    else:
                        risk_invest_proportion[i] = np.random.normal(0.15, 0.08)
        else:
            # Ramsey: age-independent
            mean = 0.50 if country == 'US' else 0.30
            std = 0.10 if country == 'US' else 0.10
            risk_invest_proportion = np.random.normal(mean, std, size=households_n)

        return np.clip(risk_invest_proportion, 0, 1)

    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, bank_action=None, env=None,
                   firm_action=None):
        if self.agent_name in ["government", "pension_gov", "tax_gov", "central_bank_gov"]:
            # Identify the specific government type and its source
            if self.agent_name == "government":
                gov_type = self.envs.government.type
                gov_agent = env.government
            elif self.agent_name == "pension_gov":
                gov_type = "pension"
                gov_agent = env.pension_gov
                tax_obj = env.tax_gov  # debt info may come from tax_gov
            elif self.agent_name == "tax_gov":
                gov_type = "tax"
                gov_agent = env.tax_gov
            elif self.agent_name == "central_bank_gov":
                gov_type = "central_bank"
                gov_agent = env.central_bank_gov
            else:
                raise ValueError(f"Invalid agent_name: {self.agent_name}. Expected one of ['government', 'pension_gov', 'tax_gov', 'central_bank_gov']")
        
            if gov_type == "central_bank":
                pi_t = global_obs_tensor[4]  # Current inflation rate
                g_t = global_obs_tensor[7]  # Current GDP growth rate
                g_star = 0.05  # Target GDP growth rate (5%)
                pi_star = 0.02  # Target inflation rate
                r_star = 0.02  # Natural rate
    
                phi_pi = 1.5
                phi_g = 0.5
    
                interest_rate = r_star + pi_t + phi_pi * (pi_t - pi_star) + phi_g * (g_t - g_star)
                reserve_ratio = gov_agent.reserve_ratio
    
                action = np.array([interest_rate.cpu().numpy(), reserve_ratio])
                noise_scale = np.array([0.002, 0.02])
                action += np.random.normal(0, noise_scale)
                return action
            # Pension Logic
            elif gov_type == "pension":   ## IMF retirement adjustment rule
                Bt = getattr(env, 'tax_gov', gov_agent).Bt_next
                GDP = getattr(env, 'tax_gov', gov_agent).GDP
                debt_GDP = Bt / GDP
                phi_RA = 0.2
                phi_gamma = 0.1
                debt_ratio_upper = 0.6
            
                retire_age = gov_agent.retire_age
                contrib_rate = gov_agent.contribution_rate
            
                new_retire_age = retire_age + max(debt_GDP - debt_ratio_upper, 0) * phi_RA
                new_contrib = contrib_rate + phi_gamma * max(debt_GDP - debt_ratio_upper, 0)
            
                retire_age = int(np.clip(new_retire_age, 55, 75))
                contribution_rate = np.clip(new_contrib, 0.05, 0.3)
                return np.array([retire_age, contribution_rate])

            elif gov_type == "tax":
                # base_action = np.array([0.263, 0.049, 0., 0., 0.189])
                predefined_cfg = copy.deepcopy(gov_agent.entity_args.params)
                tau = predefined_cfg.tau
                xi = predefined_cfg.xi
                tau_a = predefined_cfg.tau_a
                xi_a = predefined_cfg.xi_a
                Gt_prob = predefined_cfg.Gt_prob

                base_action = [tau, xi, tau_a, xi_a, Gt_prob]
    
                if env.market.firm_n > 1:
                    subsidy_weights = np.ones(env.market.firm_n) / env.market.firm_n
                    action = np.concatenate([base_action, subsidy_weights])
                else:
                    action = np.array(base_action)
                    
                # noise_scale = np.ones_like(action) * 0.01
                #
                # action += np.random.normal(0, noise_scale)
                return action


        elif self.agent_name == "households":
            household_type = self.envs.households.type
            country = 'China'   # Choose from "US" or "China"; the rules are based on statistical data.
            households_n = len(private_obs_tensor)
            action = np.random.randn(households_n, self.envs.households.action_dim)
            if household_type in ['OLG', 'OLG_risk_invest']:
                age = env.households.age
            else:
                age = None
            saving = self.generate_saving_proportion(age, households_n, country=country)
            action[:, 0] = saving.flatten()
            # Generate risk investment ratio for risk_invest types
            if household_type in ['OLG_risk_invest', 'ramsey_risk_invest']:
                risk = self.generate_risk_investment_proportion(age, households_n, country=country)
                action[:, 2] = risk.flatten()
            action[:, 1] = np.clip(action[:, 1], 0, 1)  # Ensure [0,1] for second dimension
            return action

        elif self.agent_name == "market":
            return np.random.randn(self.envs.market.firm_n, self.envs.market.action_dim)

        elif self.agent_name == "bank":
            return np.random.randn(self.envs.bank.action_dim)


    def train(self, transition):
        return 0, 0

    def save(self, dir_path, step=0):
        pass
