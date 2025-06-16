import os

import copy
import torch
import numpy as np
from agents.models import mlp_net
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.utils.rnn as rnn_utils


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)


class ppo_agent:
    def __init__(self, envs, args, agent_name="household"):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        
        env_agent_name = "households" if agent_name == "household" else agent_name

        self.agent = getattr(self.envs, env_agent_name)
        self.obs_dim = self.agent.observation_space.shape[0]
        self.action_dim = self.agent.action_space.shape[-1]

        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.net = mlp_net(state_dim=self.obs_dim, num_actions=self.action_dim).to(self.device)

        self.load_exist_policy = False  # choose from "test" or "train". if test, get action from trained ppo policy.

        if self.load_exist_policy == True and agent_name == "household":
            if "OLG" in self.envs.households.type:
                self.net.load_state_dict(torch.load("agents/models/trained_policy/ppo_OLG/ppo_net.pt",
                                                    weights_only=True))  # 30,31   # 这个 policy 不能风投，当时没设置
            elif "ramsey" in self.envs.households.type:
                self.net = mlp_net(state_dim=self.obs_dim, num_actions=3).to(
                    self.device)  # 训练的是 action_dim =3,如果不要最后一维，可以截掉
                self.net.load_state_dict(
                    torch.load("agents/models/trained_policy/ppo_Ramsey/ppo_net.pt", weights_only=True))
        elif self.load_exist_policy == True and agent_name == "government":
            if self.envs.government.type == "tax":
                self.net.load_state_dict(
                    torch.load("agents/models/bc_ppo/100/gdp/run8/ppo_net.pt", weights_only=True))
            # elif self.envs.government.type == "central_bank":
            #     self.net.load_state_dict(
            #         torch.load("agents/models/bc_ppo/100/gdp/run15/ppo_net.pt", weights_only=True))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.p_lr, eps=1e-5)
        lambda_function = lambda epoch: 0.97 ** (epoch // 10)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_function)
        self.on_policy = True
        self.state_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.step_counter = 0
        self.max_warmup_steps = 1000
        self.normalizer_applied = False

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_output = torch.tensor(np.array(advantage_list), dtype=torch.float)
        # batch adv norm
        norm_advantage = (advantage_output - torch.mean(advantage_output)) / torch.std(advantage_output)
        return norm_advantage

    def train(self, transition_dict):
        sum_loss = torch.tensor([0., 0.], dtype=torch.float32).to(self.device)
        global_obses = torch.tensor(np.array(transition_dict['global_obs']), dtype=torch.float32).to(self.device)
        if self.agent_name == "pension_gov":
            gov_actions = torch.tensor(np.array(transition_dict['pension_gov_action']), dtype=torch.float32).to(self.device)

        elif self.agent_name == "central_bank_gov":
            gov_actions = torch.tensor(np.array(transition_dict['central_bank_gov_action']), dtype=torch.float32).to(self.device)
        else:
            gov_actions = torch.tensor(np.array(transition_dict['gov_action']), dtype=torch.float32).to(
                self.device)

        gov_rewards = torch.tensor(np.array(transition_dict['gov_reward']), dtype=torch.float32).to(
            self.device).unsqueeze(-1)

        next_global_obses = torch.tensor(np.array(transition_dict['next_global_obs']), dtype=torch.float32).to(
            self.device)
        inverse_dones = torch.tensor([x - 1 for x in np.array(transition_dict['done'])], dtype=torch.float32).to(
            self.device).unsqueeze(-1)

        # private_obses = [torch.tensor(obs, dtype=torch.float32) for obs in transition_dict['private_obs']]
        # house_actions = [torch.tensor(obs, dtype=torch.float32) for obs in transition_dict['house_action']]
        # house_rewards = [torch.tensor(obs, dtype=torch.float32) for obs in transition_dict['house_reward']]
        # next_private_obses = [torch.tensor(obs, dtype=torch.float32) for obs in transition_dict['next_private_obs']]
        #
        # private_obses_tensor = rnn_utils.pad_sequence(private_obses, batch_first=True)
        # house_actions_tensor = rnn_utils.pad_sequence(house_actions, batch_first=True)
        # house_rewards_tensor = rnn_utils.pad_sequence(house_rewards, batch_first=True)
        # next_private_obses_tensor = rnn_utils.pad_sequence(next_private_obses, batch_first=True)

        # households_n = len(private_obses_tensor[0])
        if self.agent_name in ("government", "pension_gov", "tax_gov", "central_bank_gov"):
            obs_tensor = global_obses
            action_tensor = self.inverse_action_wrapper(gov_actions)
            reward_tensor = gov_rewards
            next_obs_tensor = next_global_obses
        # elif self.agent_name == "household":
        #     obs_tensor = private_obses_tensor
        #     action_tensor = house_actions_tensor
        #     reward_tensor = house_rewards_tensor
        #     next_obs_tensor = next_private_obses_tensor
        #     inverse_dones = inverse_dones.unsqueeze(-1).repeat(1, households_n, 1)
        else:
            obs_tensor, action_tensor, reward_tensor, next_obs_tensor = None, None, None, None

        next_value, next_pi = self.net(next_obs_tensor)
        td_target = reward_tensor + self.args.gamma * next_value * inverse_dones
        value, pi = self.net(obs_tensor)
        td_delta = td_target - value
        advantage = self.compute_advantage(self.args.gamma, self.args.tau, td_delta.cpu()).to(self.device)
        mu, std = pi
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(action_tensor)

        for i in range(self.args.update_each_epoch):
            value, pi = self.net(obs_tensor)
            mu, std = pi
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(action_tensor)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(value, td_target.detach()))
            total_loss = actor_loss + self.args.vloss_coef * critic_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()
            sum_loss[0] += actor_loss
            sum_loss[1] += critic_loss

        self.scheduler.step()
        return sum_loss[0], sum_loss[1]

    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None,env=None):
        if self.agent_name in ("government", "tax_gov", "pension_gov", "central_bank_gov"):
            obs_tensor = global_obs_tensor.reshape(-1, self.obs_dim)
        else:
            obs_tensor = private_obs_tensor
        # === 1. Running mean/std update ===
        if self.step_counter < self.max_warmup_steps:
            obs_np = obs_tensor.detach().cpu().numpy()
            self.state_rms.update(obs_np)
            self.step_counter += 1

        # === 2. Inject normalizer after warmup ===
        if (not self.normalizer_applied) and (self.step_counter >= self.max_warmup_steps):
            self.net.set_normalizer(self.state_rms.mean, self.state_rms.std)
            self.normalizer_applied = True

        _, pi = self.net(obs_tensor)
        mu, sigma = pi
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        if self.agent_name == "government" and self.envs.government.type == "tax" or self.agent_name == "tax_gov":
            action[0][2] = 0
            action[0][3] = 0

        if self.agent_name in ("government", "tax_gov", "pension_gov", "central_bank_gov"):
            return self.gov_action_wrapper(action.cpu().numpy().flatten())
        else:
            return action.cpu().numpy()

    def gov_action_wrapper(self, gov_action):
        return self.agent.real_action_min + (self.agent.real_action_max - self.agent.real_action_min) * gov_action

    def inverse_action_wrapper(self, action):
        if action.is_cuda:
            action_np = action.cpu().numpy()  # 将张量从 GPU 移动到 CPU 并转换为 numpy 数组
        else:
            action_np = action.numpy()  # 如果 action 已经在 CPU 上，则直接转换为 numpy 数组

        result_np = (action_np - self.agent.real_action_min) / (
                self.agent.real_action_max - self.agent.real_action_min)

        # 将计算结果转换回 PyTorch 的 tensor，并根据原始 action 是否在 GPU 上决定是否要将结果放回 GPU
        result_tensor = torch.tensor(result_np, dtype=action.dtype)

        if action.is_cuda:
            return result_tensor.cuda()  # 如果原始 action 在 GPU 上，则将结果也放回 GPU
        else:
            return result_tensor  # 否则保持在 CPU 上

    def save(self, dir_path):
        torch.save(self.net.state_dict(), str(dir_path) + '/ppo_net.pt')
