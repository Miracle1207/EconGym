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

        if self.agent_name == "government":
            if type is None:
                raise ValueError(
                    "agent_type must be specified for government agent (e.g., 'central_bank', 'tax', 'pension')")
                self.government_agent = self.envs.government[type]
                self.obs_dim = self.government_agent.observation_space.shape[0]
                self.action_dim = self.government_agent.action_space.shape[0]
        elif self.agent_name == "bank":
            self.obs_dim = self.envs.bank.observation_space.shape[0]
            self.action_dim = self.envs.bank.action_space.shape[-1]
        elif self.agent_name == "market":
            self.obs_dim = self.envs.market.observation_space.shape
            self.action_dim = self.envs.market.action_space.shape[-1]
        else:
            raise ValueError(f"LLM agent does not support agent_name '{self.agent_name}'.")

        # self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.client = OpenAI(api_key="sk-9f0862fb6328446f99cdca6e84ecf4ae", base_url="https://api.deepseek.com")

        self.device = "cuda" if getattr(self.args, 'cuda', False) else "cpu"
        self.on_policy = True

    def get_action(self, obs_tensor):
        """Produce actions for the configured role using an LLM."""
        obs_np = obs_tensor.cpu().numpy() if isinstance(obs_tensor, torch.Tensor) else np.array(obs_tensor)

        if self.agent_name == "government":
            if self.agent_type == "central_bank":
                prompt = build_central_bank_prompt(obs_np)
            elif self.agent_type == "pension":
                prompt = build_pension_prompt(obs_np)
            else:
                prompt = build_tax_prompt(obs_np)
        elif self.agent_name == "bank":
            if self.agent_type == "non_profit":
                return np.random.randn(self.action_dim)
            prompt = build_bank_prompt(obs_np)
        elif self.agent_name == "market":
            if self.agent_type == "perfect":
                firm_n = len(obs_tensor)
                return np.random.randn(firm_n, self.action_dim)
            prompt = build_market_prompt(obs_np)
        else:
            raise ValueError(f"Unsupported agent_name '{self.agent_name}' for LLM agent")

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            print(f"{self.agent_name} Get LLM  Response Successfully")

            gpt_output = response.choices[0].message.content
            decision = self.extract_json(gpt_output)
            if decision is None:
                raise ValueError("Failed to parse GPT output.")

            if self.agent_name == "government":
                if self.agent_type == "central_bank":
                    base_interest_rate = float(decision["base_interest_rate"])
                    reserve_ratio = float(decision["reserve_ratio"])
                    return np.array([base_interest_rate, reserve_ratio])
                elif self.agent_type == "pension":
                    retire_age = float(decision["retire_age"])
                    contribution_rate = float(decision["contribution_rate"])
                    return np.array([retire_age, contribution_rate])
                else:
                    Gt_prob = float(decision["Gt_prob"])
                    tau = float(decision["tau"])
                    xi = float(decision["xi"])
                    return np.array([tau, xi, 0, 0, Gt_prob])
            elif self.agent_name == "bank":
                lending_rate = float(decision["lending_rate"])
                deposit_rate = float(decision["deposit_rate"])
                return np.array([lending_rate, deposit_rate])
            elif self.agent_name == "market":
                if isinstance(decision, list):
                    actions = [[float(item["price"]), float(item["wage"])] for item in decision]
                    return np.array(actions, dtype=np.float32)
                else:
                    raise ValueError("Market decision must be a list of {price, wage}")

        except Exception as e:
            print("Error calling GPT or parsing result:", e)
            if self.agent_name == "government":
                if self.agent_type == "central_bank":
                    return np.array(obs_np[-2:])
                elif self.agent_type == "pension":
                    return np.array(obs_np[-2:])
                else:
                    return np.array([0.2, 0.2, 0.0, 0.0, float(obs_np[-1])])
            elif self.agent_name == "bank":
                base_rate = float(obs_np[0]) if len(obs_np) > 0 else 0.03
                return np.array([base_rate + 0.02, base_rate - 0.005])
            elif self.agent_name == "market":
                firm_n = obs_np.shape[0] if obs_np.ndim == 2 else 1
                return np.zeros((firm_n, 2), dtype=np.float32)

    def save(self, dir_path):
        pass

    def extract_json(self, gpt_output):
        """
        尝试从 GPT 的输出文本中解析出 JSON 格式数据。
        为了尽可能提高成功率，本函数会依次采用多种方式进行提取和解析：
            1. 整体尝试解析
            2. 提取 ```json ...``` 代码块
            3. 提取任意 ```...``` 代码块
            4. 从第一处 '{' 到最后一处 '}' 截取并解析
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
