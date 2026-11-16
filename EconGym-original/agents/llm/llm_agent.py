import copy
import json
import os
import re

import numpy as np
import torch
from openai import OpenAI

from .prompts import (
    build_bank_prompt,
    build_central_bank_prompt,
    build_market_prompt,
    build_pension_prompt,
    build_tax_prompt,
)


def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class llm_agent:
    def __init__(self, envs, args, type=None, agent_name="government"):
        """
        LLM-driven policy for multiple roles: government, bank, market.
        - agent_name: "government" | "bank" | "market"
        - type: for government, one of {"central_bank", "tax", "pension"}
        """
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        self.agent_type = type
        self.llm_name = args.llm_name
        self.device = "cuda" if getattr(self.args, 'cuda', False) else "cpu"
        self.on_policy = True

        # Map agent_name to corresponding environment object
        if self.agent_name == "government":
            if type is None:
                raise ValueError(
                    "agent_type must be specified for government agent (e.g., 'central_bank', 'tax', 'pension')")
            agent = self.envs.government[type]
            self.gov_task = agent.gov_task
        elif self.agent_name == "bank":
            agent = self.envs.bank
        elif self.agent_name == "market":
            agent = self.envs.market
        else:
            raise ValueError(f"LLM agent does not support agent_name '{self.agent_name}'.")

        obs_shape = agent.observation_space.shape
        self.obs_dim = obs_shape[0] if len(obs_shape) > 0 else obs_shape
        self.action_dim = agent.action_space.shape[-1]

        # Get API key: prefer environment variable
        api_key = os.environ.get("OPENAI_API_KEY") or getattr(args, "api_key", None)
        if not api_key:
            raise ValueError("API key not found. Please set `self.api_key` or `OPENAI_API_KEY` environment variable.")

        # Choose client based on LLM name
        llm_name = self.llm_name.lower()
        if "deepseek" in llm_name:
            # DeepSeek uses its own API endpoint
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        elif "gpt" in llm_name:
            # GPT series uses default OpenAI endpoint
            self.client = OpenAI(api_key=api_key)
        else:
            # Other LLMs need to be configured manually
            raise ValueError(f"Unsupported LLM name '{self.llm_name}'. Please configure your own client.")


    def build_prompt(self, agent_name, agent_type, obs_np):
        """Build the LLM prompt based on agent role and type."""
        if agent_name == "government":
            if agent_type == "central_bank":
                return build_central_bank_prompt(obs_np)
            elif agent_type == "pension":
                return build_pension_prompt(obs_np, objective=self.gov_task)
            else:
                return build_tax_prompt(obs_np, objective=self.gov_task)
    
        elif agent_name == "bank":
            return build_bank_prompt(obs_np)
    
        elif agent_name == "market":
            return build_market_prompt(obs_np)
    
        else:
            raise ValueError(f"Unsupported agent_name '{agent_name}' for LLM agent")

    def parse_action(self, agent_name, agent_type, decision, obs_np):
        """Parse the LLM decision into structured actions with fallback."""
        try:
            if agent_name == "government":
                if agent_type == "central_bank":
                    return np.array([float(decision["base_interest_rate"]), float(decision["reserve_ratio"])])
                elif agent_type == "pension":
                    return np.array([float(decision["retire_age"]), float(decision["contribution_rate"])])
                else:  # tax
                    return np.array([
                        float(decision["tau"]),
                        float(decision["xi"]),
                        float(decision["tau_a"]),
                        float(decision["xi_a"]),
                        float(decision["Gt_prob"])
                    ])
        
            elif agent_name == "bank":
                return np.array([float(decision["lending_rate"]), float(decision["deposit_rate"])])
        
            elif agent_name == "market":
                if isinstance(decision, list):
                    return np.array([[float(item["price"]), float(item["wage"])] for item in decision],
                                    dtype=np.float32)
                else:
                    raise ValueError("Market decision must be a list of {price, wage}")
    
        except Exception as e:
            print("Error parsing decision, fallback to heuristic:", e)
        
            # Fallback strategies
            if agent_name == "government":
                if agent_type in ["central_bank", "pension"]:
                    return np.array(obs_np[-2:])
                else:  # tax fallback
                    return np.array([0., 0., 0., 0., float(obs_np[-1])])
        
            elif agent_name == "bank":
                base_rate = float(obs_np[0]) if len(obs_np) > 0 else 0.03
                return np.array([base_rate + 0.02, base_rate - 0.005])
        
            elif agent_name == "market":
                firm_n = obs_np.shape[0] if obs_np.ndim == 2 else 1
                return np.zeros((firm_n, 2), dtype=np.float32)
            
    def extract_json(self, gpt_output):
        """
        Attempt to parse JSON-formatted data from the GPT output text.
        To maximize the success rate, this function tries multiple extraction and parsing strategies in sequence:
            1. Directly attempt to parse the entire text
            2. Extract the ```json ...``` code block
            3. Extract any ```...``` code block
            4. Capture the substring from the first '{' to the last '}' and parse it
        """
        try:
            return json.loads(gpt_output)
        except Exception:
            pass

        pattern_json_block = r"```json\s*([\s\S]*?)\s*```"
        match_json_block = re.search(pattern_json_block, gpt_output, re.IGNORECASE)
        if match_json_block:
            json_str = match_json_block.group(1).strip()
            try:
                return json.loads(json_str)
            except Exception:
                pass

        pattern_code_block = r"```([\s\S]*?)```"
        match_code_block = re.search(pattern_code_block, gpt_output)
        if match_code_block:
            json_str = match_code_block.group(1).strip()
            try:
                return json.loads(json_str)
            except Exception:
                pass

        start_index = gpt_output.find("{")
        end_index = gpt_output.rfind("}")
        if start_index != -1 and end_index != -1 and start_index < end_index:
            possible_json_str = gpt_output[start_index:end_index + 1].strip()
            try:
                return json.loads(possible_json_str)
            except Exception:
                pass

        return None

    def get_action(self, obs_tensor):
        """Produce actions for the configured role using an LLM."""
        # Handle special random-action cases
        if self.agent_name == "bank" and self.agent_type == "non_profit":
            return np.random.randn(self.action_dim)
        elif self.agent_name == "market" and self.agent_type == "perfect":
            firm_n = len(obs_tensor)
            return np.random.randn(firm_n, self.action_dim)
    
        # Convert obs to numpy
        obs_np = obs_tensor.cpu().numpy() if isinstance(obs_tensor, torch.Tensor) else np.array(obs_tensor)
    
        # Build prompt
        prompt = self.build_prompt(self.agent_name, self.agent_type, obs_np)
    
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            gpt_output = response.choices[0].message.content
            decision = self.extract_json(gpt_output)
        
            if decision is None:
                raise ValueError("Failed to parse GPT output.")
        
            print(f"LLM-driven {self.agent_name} response successful: {decision}")
            return self.parse_action(self.agent_name, self.agent_type, decision, obs_np)
    
        except Exception as e:
            print("Error calling GPT, fallback to heuristic:", e)
            return self.parse_action(self.agent_name, self.agent_type, {}, obs_np)  # empty decision → fallback

    def train(self, transition_dict):
        """
        Train the agent using collected transition data.

        Parameters
        ----------
        transition_dict : dict
            A dictionary containing transitions. Typically includes:
            {
                "obs": ...,          # current state/observation
                "action": ...,       # action taken
                "reward": ...,       # reward received
                "next_obs": ...,     # next state
                "done": ...          # termination flag
            }

        Notes
        -----
        - This method is intentionally left unimplemented.
        - Users should define how to optimize the policy or value function.
        - A common design is to use a memory module to store
          transitions and sample mini-batches for training.
        - Example usage:
            1. Store transition_dict into memory.
            2. Sample a batch from memory.
            3. Sample few shot from memory to optimize decision-making.
        """
        pass

    def save(self, dir_path):
        pass
