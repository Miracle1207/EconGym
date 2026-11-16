import torch.nn.init as init

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class StackelbergMlp(nn.Module):
    def __init__(self, state_dim, follower_action_dim, hidden_dim, action_dim):
        super(StackelbergMlp, self).__init__()
        self.mu_emb = nn.Linear(follower_action_dim, hidden_dim)
        self.std_emb = nn.Linear(follower_action_dim, hidden_dim)
        self.obs_emb = nn.Linear(state_dim, hidden_dim)

        self.fc1_v = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_a = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2_a = nn.Linear(hidden_dim, hidden_dim)
        self.sigma_log = nn.Parameter(torch.zeros(1, action_dim))
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)
        orthogonal_init(self.fc1_v)
        orthogonal_init(self.fc2_v)
        orthogonal_init(self.fc1_a)
        orthogonal_init(self.fc2_a)
        orthogonal_init(self.value)
        orthogonal_init(self.mu_emb)
        orthogonal_init(self.std_emb)
        orthogonal_init(self.obs_emb)
        orthogonal_init(self.action_mean, gain=0.01)

    def forward(self, x, f_mu, f_std):
        mu_emb = self.mu_emb(f_mu)
        std_emb = self.std_emb(f_std)
        obs_emb = self.obs_emb(x)
        x_concat = torch.concat([obs_emb, mu_emb, std_emb], dim=-1)
        x_v = torch.tanh(self.fc1_v(x_concat))
        x_v = torch.tanh(self.fc2_v(x_v))
        state_value = self.value(x_v)

        x_a = torch.tanh(self.fc1_a(x_concat))
        x_a = torch.tanh(self.fc2_a(x_a))
        mean = self.action_mean(x_a)
        sigma_log = self.sigma_log.expand_as(mean)
        sigma = torch.exp(sigma_log)
        pi = (mean, sigma)

        return state_value, pi


class mlp_net(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super(mlp_net, self).__init__()
        self.use_norm = False
        self.fc1_v = nn.Linear(state_dim, hidden_dim)
        self.fc2_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_a = nn.Linear(state_dim, hidden_dim)
        self.fc2_a = nn.Linear(hidden_dim, hidden_dim)
        # check the type of distribution
        self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))
        self.action_mean = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)
        orthogonal_init(self.fc1_v)
        orthogonal_init(self.fc2_v)
        orthogonal_init(self.fc1_a)
        orthogonal_init(self.fc2_a)
        orthogonal_init(self.value)
        orthogonal_init(self.action_mean, gain=0.01)

    def set_normalizer(self, mean, std):
        self.use_norm = True
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
        if next(self.parameters()).is_cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def forward(self, x):
        if self.use_norm:
            x = (x - self.mean) / (self.std + 1e-8)
        x_v = torch.tanh(self.fc1_v(x))
        x_v = torch.tanh(self.fc2_v(x_v))
        state_value = self.value(x_v)

        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))
        mean = torch.tanh(self.action_mean(x_a))  # in (-1,1)
        # mean = torch.sigmoid(self.action_mean(x_a))  # in (0,1)
        
        # Handle both 1D and 2D inputs properly
        if mean.dim() == 1:
            # For 1D input (e.g., government agent), sigma_log is [1, num_actions]
            # We need to squeeze the first dimension to match mean's shape
            sigma_log = self.sigma_log.squeeze(0)  # Remove the batch dimension
        else:
            # For 2D input (e.g., households with batch), expand normally
            sigma_log = self.sigma_log.expand(mean.shape)
        
        sigma = torch.exp(sigma_log)  # Convert log to actual sigma value

        pi = (mean, 0.05 * sigma)

        return state_value, pi


class aie_mlp_net(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128, lstm_hidden_dim=128):
        super(aie_mlp_net, self).__init__()
        self.fc1_v = nn.Linear(state_dim, hidden_dim)
        self.fc2_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_a = nn.Linear(state_dim, hidden_dim)
        self.fc2_a = nn.Linear(hidden_dim, hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self._init_lstm_weights(self.lstm)
        # Updated dimensions for action_mean and value layers
        self.action_mean = nn.Linear(lstm_hidden_dim, num_actions)
        self.value = nn.Linear(lstm_hidden_dim, 1)
        # Other layers
        self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))
        # Orthogonal initialization with modified gains
        self._init_weights()

    def _init_lstm_weights(self, lstm):
        # Initialize weights for LSTM
        for name, param in lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def _init_weights(self):
        # Adjust gain based on the activation function
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(self.fc1_v.weight, gain)
        nn.init.orthogonal_(self.fc2_v.weight, gain)
        nn.init.orthogonal_(self.fc1_a.weight, gain)
        nn.init.orthogonal_(self.fc2_a.weight, gain)
        nn.init.orthogonal_(self.value.weight, gain)
        nn.init.orthogonal_(self.action_mean.weight, 0.01)  # Specific gain for this layer

    def forward(self, x):
        # Process with fully connected layers using ReLU
        x_v = torch.tanh(self.fc1_v(x))
        x_v = torch.tanh(self.fc2_v(x_v))
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))
        # Apply LSTM layer
        x_v, _ = self.lstm(x_v)
        x_a, _ = self.lstm(x_a)

        state_value = self.value(x_v)
        mean = self.action_mean(x_a)
        sigma_log = self.sigma_log.expand_as(mean)
        sigma = torch.exp(sigma_log).clamp(min=1e-3, max=50)  # Add clamping to avoid extreme values
        pi = (mean, sigma)
        return state_value, pi


class PredictNet(nn.Module):
    def __init__(self, state_dim, follower_action_dim, hidden_dim, action_dim):
        super(PredictNet, self).__init__()
        self.obs_emb = nn.Linear(state_dim, hidden_dim)
        self.action_emb = nn.Linear(action_dim, hidden_dim)
        self.concat_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mean = nn.Linear(hidden_dim, follower_action_dim)
        self.log_std = nn.Linear(hidden_dim, follower_action_dim)

    def forward(self, obs, action):
        x = F.relu(self.obs_emb(obs))
        y = F.relu(self.action_emb(action))
        concate_x = torch.concat([x, y], dim=-1)
        out = F.relu(self.concat_layer(concate_x))
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return (mean, torch.exp(log_std))


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.use_norm = False
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def set_normalizer(self, mean, std):
        self.use_norm = True
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        if self.use_norm:
            x = (x - self.mean) / (self.std + 1e-8)
        x = F.relu(self.fc1(x))
        # return torch.sigmoid(self.fc2(x))  # [0, 1]
        return torch.tanh(self.fc2(x))  # [-1, 1]


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class BiQValueNet(torch.nn.Module):
    def __init__(self, gov_state_dim, gov_action_dim, mean_state_dim, mean_action_dim, hidden_dim, hidden_dim_1=64):
        super(BiQValueNet, self).__init__()
        self.fc1 = nn.Linear(gov_state_dim + gov_action_dim + mean_state_dim + mean_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, sl, al, mean_sf, mean_af):
        cat = torch.cat([sl, al, mean_sf, mean_af], dim=-1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class MFActor(nn.Module):
    def __init__(self, input_dims, leader_action_dim, action_dims, hidden_size):
        super(MFActor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.gov_action_emb = nn.Linear(leader_action_dim, hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, action_dims)
        orthogonal_init(self.fc1)
        orthogonal_init(self.gov_action_emb)
        orthogonal_init(self.fc2)

    def forward(self, obs, gov_action):
        x = F.relu(self.fc1(obs))
        y = F.relu(self.gov_action_emb(gov_action))
        z = torch.cat((x, y), dim=-1)
        return torch.tanh(self.fc2(z))


class MFCritic(nn.Module):
    def __init__(self, obs_dims, gov_action_dim, action_dim, mean_state_dim, mean_action_dim, hidden_size,
                 hidden_size_1=64):
        super(MFCritic, self).__init__()
        self.fc1 = nn.Linear(obs_dims, hidden_size_1)
        self.gov_action_emb = nn.Linear(gov_action_dim, hidden_size_1)
        self.house_action_emb = nn.Linear(action_dim, hidden_size_1)
        self.mean_action_emb = nn.Linear(mean_action_dim, hidden_size_1)
        self.mean_state_emb = nn.Linear(mean_state_dim, hidden_size_1)
        self.q_value_fc1 = nn.Linear(hidden_size_1 * 5, hidden_size)
        self.q_value_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

    def forward(self, obs, gov_action, action, mean_state, mean_action):
        obs_emb = F.relu(self.fc1(obs))
        gov_action_emb = F.relu(self.gov_action_emb(gov_action))
        action_emb = F.relu(self.house_action_emb(action))
        mean_state_emb = F.relu(self.mean_action_emb(mean_state))
        mean_action_emb = F.relu(self.mean_action_emb(mean_action))

        inputs_emb = torch.cat([obs_emb, gov_action_emb, action_emb, mean_state_emb, mean_action_emb], dim=-1)
        emb = F.relu(self.q_value_fc1(inputs_emb))
        emb = F.relu(self.q_value_fc2(emb))
        output = self.q_value(emb)
        return output


class MFCritic_Single(nn.Module):
    def __init__(self, obs_dims, action_dim, mean_state_dim, mean_action_dim, hidden_size,
                 hidden_size_1=64):
        super(MFCritic_Single, self).__init__()
        self.fc1 = nn.Linear(obs_dims, hidden_size_1)
        self.house_action_emb = nn.Linear(action_dim, hidden_size_1)
        self.mean_action_emb = nn.Linear(mean_action_dim, hidden_size_1)
        self.mean_state_emb = nn.Linear(mean_state_dim, hidden_size_1)
        self.q_value_fc1 = nn.Linear(hidden_size_1 * 4, hidden_size)
        self.q_value_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

    def forward(self, obs, action, mean_state, mean_action):
        obs_emb = F.relu(self.fc1(obs))
        action_emb = F.relu(self.house_action_emb(action))
        mean_state_emb = F.relu(self.mean_action_emb(mean_state))
        mean_action_emb = F.relu(self.mean_action_emb(mean_action))

        inputs_emb = torch.cat([obs_emb, action_emb, mean_state_emb, mean_action_emb], dim=-1)
        emb = F.relu(self.q_value_fc1(inputs_emb))
        emb = F.relu(self.q_value_fc2(emb))
        output = self.q_value(emb)
        return output


class CloneModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(CloneModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_mean = nn.Linear(hidden_size, output_size)
        self.fc_std = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = torch.sigmoid(self.fc_mean(x))  # Apply sigmoid activation
        log_std = torch.clamp(self.fc_std(x), min=-20, max=2)
        std = torch.exp(log_std)  # Ensure std is positive
        return mean, std


class Critic(nn.Module):
    def __init__(self, input_dims, hidden_size, action_dims=None):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size) if action_dims is None else nn.Linear(input_dims + action_dims,
                                                                                            hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, action=None):
        inputs = torch.cat([obs, action], dim=-1) if action is not None else obs
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output


class SharedCritic(nn.Module):  # Q(s, a_g, a_h, \bar{a_h})
    def __init__(self, state_dim, hou_action_dim, hidden_size, num_agent):
        super(SharedCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + hou_action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, hou_action):
        global_state = global_state.unsqueeze(1)
        gov_action = gov_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)

        inputs = torch.cat([n_global_obs, private_state, n_gov_action, hou_action], dim=-1)  # 修改维度
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.1)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size, log_std_min, log_std_max):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        # self.gru = nn.GRU(128, 256, 1, batch_first=True)
        self.fc2 = nn.Linear(128, 128)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(128, action_dims)
        self.log_std = nn.Linear(128, action_dims)
        self.log_std_min = log_std_min
        self.log_std_max = 1
        self.mean_max = 1
        self.mean_min = -1

    def forward(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        mean = torch.clamp(mean, min=self.mean_max, max=self.mean_min)

        return (mean, torch.exp(log_std))


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return out


# for households
class SharedAgent(nn.Module):
    def __init__(self, input_size, output_size, num_agent, log_std_min, log_std_max):
        super(SharedAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.gru = GRU(64, 128, 1, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(64, output_size)
        self.log_std = nn.Linear(64, output_size)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, update=False):
        if update == True:
            global_state = global_state.unsqueeze(1)
            gov_action = gov_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        inputs = torch.cat([n_global_obs, private_state, n_gov_action], dim=-1)
        out = self.fc1(inputs)
        out = self.gru(out)
        out = self.fc2(out)
        out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))


class mlp(nn.Module):
    def __init__(self, input_size, output_size, num_agent, log_std_min, log_std_max):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.gru = GRU(64, 128, 1, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(64, output_size)
        self.log_std = nn.Linear(64, output_size)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent

    def forward(self, global_state, private_state, update=False):
        n_global_obs = global_state.repeat(1, self.num_agent, 1)

        inputs = torch.cat([n_global_obs, private_state], dim=-1)
        out = self.fc1(inputs)
        out = self.gru(out)
        out = self.fc2(out)
        out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))


class MFSharedAgent(nn.Module):
    def __init__(self, input_size, gov_action_dim, house_action_dim, num_agent, log_std_min, log_std_max):
        super(MFSharedAgent, self).__init__()
        self.fc1 = nn.Linear(input_size + gov_action_dim + house_action_dim, 64)
        self.gru = GRU(64, 128, 1, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(64, house_action_dim)
        self.log_std = nn.Linear(64, house_action_dim)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, past_mean_house_action, update=False):
        if update == True:
            global_state = global_state.unsqueeze(1)
            gov_action = gov_action.unsqueeze(1)
            past_mean_house_action = past_mean_house_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        n_past_mean_house_action = past_mean_house_action.repeat(1, self.num_agent, 1)
        inputs = torch.cat([n_global_obs, private_state, n_gov_action, n_past_mean_house_action], dim=-1)
        out = self.fc1(inputs)
        out = self.gru(out)
        out = self.fc2(out)
        out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))


class MFSharedCritic(nn.Module):  # Q(s, a_g, a_h, \bar{a_h})
    def __init__(self, state_dim, gov_action_dim, hou_action_dim, hidden_size, num_agent):
        super(MFSharedCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + gov_action_dim + 2 * hou_action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, hou_action, mean_house_action):
        global_state = global_state.unsqueeze(1)
        gov_action = gov_action.unsqueeze(1)
        mean_house_action = mean_house_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        n_mean_house_action = mean_house_action.repeat(1, self.num_agent, 1)

        inputs = torch.cat([n_global_obs, private_state, n_gov_action, hou_action, n_mean_house_action], dim=-1)  # 修改维度
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output


class BMF_actor(nn.Module):
    def __init__(self, input_size, gov_action_dim, house_action_dim, num_agent, log_std_min, log_std_max):
        super(BMF_actor, self).__init__()
        self.fc1 = nn.Linear(input_size + gov_action_dim + house_action_dim * 2,
                             64)  # pi(observation, gov_a, top10_action, bot50_action)
        self.gru = GRU(64, 128, 1, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(64, house_action_dim)
        self.log_std = nn.Linear(64, house_action_dim)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, past_mean_house_action, update=False):
        if update == True:
            global_state = global_state.unsqueeze(1)
            gov_action = gov_action.unsqueeze(1)
            past_mean_house_action = past_mean_house_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        n_past_mean_house_action = past_mean_house_action.repeat(1, self.num_agent, 1)
        inputs = torch.cat([n_global_obs, private_state, n_gov_action, n_past_mean_house_action], dim=-1)
        out = self.fc1(inputs)
        out = self.gru(out)
        out = self.fc2(out)
        out = self.tanh(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return (mean, torch.exp(log_std))


class BMF_actor_1(nn.Module):
    def __init__(self, input_size, gov_action_dim, house_action_dim, num_agent, log_std_min, log_std_max):
        super(BMF_actor_1, self).__init__()
        self.fc1 = nn.Linear(input_size + gov_action_dim, 128)

        self.fc2 = nn.Linear(128, 128)
        self.tanh = nn.Tanh()
        self.mean = nn.Linear(128, house_action_dim)
        self.log_std = nn.Linear(128, house_action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, update=False):
        if update:
            global_state = global_state.unsqueeze(1)
            gov_action = gov_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        inputs = torch.cat([n_global_obs, private_state, n_gov_action], dim=-1)
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.relu(out)
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        mean = torch.clamp(mean, min=-1, max=1)

        return (mean, torch.exp(log_std))


class BMF_critic(nn.Module):
    def __init__(self, state_dim, gov_action_dim, hou_action_dim, hidden_size, num_agent):
        super(BMF_critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + gov_action_dim + 3 * hou_action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)
        self.num_agent = num_agent

    def forward(self, global_state, private_state, gov_action, hou_action, mean_house_action):
        global_state = global_state.unsqueeze(1)
        gov_action = gov_action.unsqueeze(1)
        mean_house_action = mean_house_action.unsqueeze(1)

        n_global_obs = global_state.repeat(1, self.num_agent, 1)
        n_gov_action = gov_action.repeat(1, self.num_agent, 1)
        n_mean_house_action = mean_house_action.repeat(1, self.num_agent, 1)

        inputs = torch.cat([n_global_obs, private_state, n_gov_action, hou_action, n_mean_house_action], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output
