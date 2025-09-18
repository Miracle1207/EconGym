
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


    def sample(self, agent_name, agent_policy, batch_size, on_policy, transition_dict=None):
        """Fetch data based on the agent's policy and mode (on-policy or off-policy)."""
        # Determine the fetch function based on on-policy flag
        fetch_function = self.fetch_on_policy_data if on_policy else self.fetch_off_policy_data
    
        # If agent_policy is a dictionary, fetch data for each sub-agent
        if isinstance(agent_policy, dict):
            return {sub_agent_name: fetch_function(agent_name, sub_agent_name, batch_size, transition_dict)
                    for sub_agent_name in agent_policy}
    
        # If it's not a dictionary, fetch data for the main agent
        return fetch_function(agent_name, agent_name, batch_size, transition_dict)
    
    def fetch_off_policy_data(self, agent_name, agent_type, batch_size, transition_dict=None):
        idxes = random.choices(range(len(self.storage)), k=batch_size)
        new_data = {key: [] for key in self.storage[0]}
        for i in idxes:
            sample = self.storage[i]
            for item_name, item_data in sample.items():
                if item_name == "done":
                    data = item_data
                else:
                    # item_data is a dictionary with agent names as keys
                    if agent_name in item_data:
                        data = item_data[agent_name]
                        if agent_name != agent_type and isinstance(data, dict) and agent_type in data:
                            data = data[agent_type]
                    else:
                        # If agent_name not found, skip this item
                        continue
                new_data[item_name].append(data)
        return new_data

    def fetch_on_policy_data(self, agent_name, agent_type, batch_size, transition_dict):
        """Fetch data for a specific agent, reducing redundancy."""
        new_data = {key: [] for key in transition_dict}
    
        for i in range(batch_size):
            for item_name, item_data in transition_dict.items():
                if item_name == "done":
                    data = item_data[i]
                else:
                    data = item_data[i][agent_name]
                    if agent_name != agent_type:
                        data = data[agent_type]
                new_data[item_name].append(data)
    
        return new_data
    


