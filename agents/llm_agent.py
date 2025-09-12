import os
import re
import copy
import torch
import numpy as np
from openai import OpenAI
import json
import re



def save_args(path, args):
    argsDict = args.__dict__
    with open(str(path) + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


class llm_agent:
    def __init__(self, envs, args, agent_name="households"):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        if agent_name == "households":
            self.obs_dim = self.envs.households.observation_space.shape[0]
            self.action_dim = self.envs.households.action_space.shape[1]
        elif agent_name == "government":
            self.obs_dim = self.envs.government.observation_space.shape[0]
            self.action_dim = self.envs.government.action_space.shape[0]
        elif agent_name == "central_bank_gov":
            self.obs_dim = self.envs.central_bank_gov.observation_space.shape[0]
            self.action_dim = self.envs.central_bank_gov.action_space.shape[0]
        elif agent_name == "tax_gov":
            self.obs_dim = self.envs.tax_gov.observation_space.shape[0]
            self.action_dim = self.envs.tax_gov.action_space.shape[0]
        elif agent_name == "pension_gov":
            self.obs_dim = self.envs.pension_gov.observation_space.shape[0]
            self.action_dim = self.envs.pension_gov.action_space.shape[0]

        else:
            print("AgentError: Please choose the correct agent name!")
            
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )

        # if use the cuda...
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.on_policy = True

    def train(self, transition_dict):
        '''you can add fine-tuning algorithms for LLM here'''
        return 0, 0

    def build_central_bank_prompt(self, global_obs, private_gov_obs):
        try:
            o = [float(x) for x in global_obs]
            p = [float(x) for x in private_gov_obs]
        
            prompt = f'''You are the policy leader of a national central bank and budget department. Based on the macro indicators below,your task is to adjust **monetary policy parameters** for the next period to **maximize social welfare**.

                    ---

                    **Your objectives**:
                    - Maximize household **consumption** (more is better)
                    - Minimize **work burden** (longer working years reduce welfare)
                    - Reduce **wealth inequality** (lower Gini)
                    - Maintain **stable GDP growth**

                    ---

                    **Macro indicators**:
                     - Mean household wealth: {o[0]:.2f}
                     - Mean household income: {o[1]:.2f}
                     - Mean education level of households: {o[2]:.2f}
                     - Households number: {o[3]:.0f}
                     - Current GDP: {o[6]:.2f}, GDP growth rate: {o[7]:.3f}, Inflation: {o[4]:.3f}
                     - Wealth Gini index: {o[5]:.3f}
                     - Projected pension burden (Bt/GDP): {o[8]:.3f}
                     - Last parameter tau : {o[9]:.3f}
                     - Last parameter xi: {o[10]:.3f}
                     - Last parameter Gt_prob: {o[11]:.3f}

                     - Last base interest rate: {p[0]:.3f}
                     - Last base reserve requirement: {p[1]:.3f}


                    ---

                    **Your reasoning should consider**:
                    1. Does a higher base interest rate encourage savings at the expense of current consumption?
                    2. Could low rates stimulate spending but threaten pension fund sustainability?
                    3. Is the current inflation level justifying an increase in interest rates?
                    4. How will changes affect borrowing costs and thus labor participation or retirement decisions?
                    5. Will the new rate support stable GDP growth and lower inequality?

                    ---
                     **Policy Parameters You Can Adjust**:
                     1. Base interest rate (`base_interest_rate`) ∈ [-0.02,0.2]
                          → Controls borrowing costs and savings incentives

                     2. Reserve requirement (`reserve_ratio`) ∈ [0,0.5]
                          → Affects banks' lending capacity and money supply


                    **Constraints**:
                    - All values must be floats.
                    - Parameter bounds strictly enforced.
                    - Max absolute change per parameter: ±0.1

                    ---

                    **Output format**:
                    Respond only with the decision in strict JSON:
                    ```json
                    {{
                      "base_interest_rate": ...    // e.g., 1.5 or 2.0
                      "reserve_ratio": ...   // e.g., 0.10
                    }}
                    ```'''
            return prompt
        except Exception as e:
            return "You need to check whether the prompt here is consistent with the global_obs set in get_obs() within env_core"

    def build_pension_prompt(self, global_obs, private_gov_obs):
        try:
            o = [float(x) for x in global_obs]
            p = [float(x) for x in private_gov_obs]
        
            prompt = f'''You are the policy leader of a national pension system and budget department. Based on the macro indicators below, your task is to adjust the **retirement age** and **contribution rate** for the next period to **maximize social welfare**.

                            ---

                            **Your objectives**:
                            - Maximize household **consumption** (more is better)
                            - Minimize **work burden** (longer working years reduce welfare)
                            - Reduce **wealth inequality** (lower Gini)
                            - Maintain **stable GDP growth**

                            ---

                            **Macro indicators**:
                            - Mean household wealth: {o[0]:.2f}
                            - Mean household income: {o[1]:.2f}
                            - Mean education level of households: {o[2]:.2f}
                            - Households number: {o[3]:.0f}
                            - Current GDP: {o[6]:.2f}, GDP growth rate: {o[7]:.3f}, Inflation: {o[4]:.3f}
                            - Wealth Gini index: {o[5]:.3f}
                            - Projected pension burden (Bt/GDP): {o[8]:.3f}
                            - Last parameter pension fund: {o[9]:.3f}
                            - Last parameter retire age: {o[10]:.3f}
                            - Last parameter contribution  ratio: {o[11]:.3f}

                            - Last parameter pension fund  ratio: {p[0]:.3f}
                            - Last parameter Retirement age  ratio: {p[1]:.3f}
                            - Last parameter Contribution rate  ratio: {p[2]:.3f}

                            ---

                            **Your reasoning should consider**:
                            1. Does increasing the retirement age help sustain the pension system without overly burdening workers?
                            2. Could higher contribution rates improve pension fund sustainability but reduce disposable income and consumption?
                            3. How will changes affect labor market participation and overall economic productivity?
                            4. Will adjustments contribute to reducing wealth inequality?
                            5. Are there any inflationary pressures that might influence these decisions?

                            ---
                            **Policy Parameters You Can Adjust**:
                            1. Retirement age (`retire_age`) ∈ [50, 75]
                            → Determines the average retirement age in the population.

                            2. Contribution rate (`contribution_rate`) ∈ [0.05, 0.2]
                            → Represents the percentage of income that goes towards pension contributions.


                            **Constraints**:
                            - All values must be floats.
                            - Parameter bounds strictly enforced.
                            - Max absolute change per parameter: ±0.1

                            ---

                            **Output format**:
                            Respond only with the decision in strict JSON:
                            ```json
                            {{
                               "retire_age": ...,   // e.g., 63.0
                               "contribution_rate": ... // e.g., 0.12
                            }}
                            ```'''
            return prompt
        except Exception:
            return "You need to check whether the prompt here is consistent with the global_obs set in get_obs() within env_core"

    def build_prompt(self, global_obs):
        try:
            o = [float(x) for x in global_obs]
        
            prompt = f'''You are the fiscal policy leader of a national taxation and budget department.
            Based on the macro indicators below,  your task is to adjust **tax policy parameters** for the next period to **maximize social welfare**.

                               ---

                               **Your objectives**:
                               - Maximize household **consumption** (more is better)
                               - Minimize **wealth inequality** (lower Gini index)
                               - Maintain **stable GDP growth**
                               - Ensure **sustainable government debt** levels
                               - Improve **public service quality** through efficient spending

                               ---

                               **Macro indicators**:
                               - Mean household wealth: {o[0]:.2f}
                               - Mean household income: {o[1]:.2f}
                               - Mean education level of households: {o[2]:.2f}
                               - Households number: {o[3]:.0f}
                               - Current GDP: {o[6]:.2f}, GDP growth rate: {o[7]:.3f}, Inflation: {o[4]:.3f}
                               - Wealth Gini index: {o[5]:.3f}
                               - Projected pension burden (Bt/GDP): {o[8]:.3f}
                               - Last parameter tau : {o[9]:.3f}
                               - Last parameter xi: {o[10]:.3f}
                               - Last parameter Gt_prob: {o[11]:.3f}


                               ---

                               **Policy Parameters You Can Adjust**:

                                1. Government spending ratio (`Gt_prob`) ∈ [0.01, 0.1]
                                 → Controls government consumption as a share of GDP

                                2. Average marginal income tax rate (`τ`) ∈ [0.01, 1]
                                 → Affects labor supply, disposable income, and work incentives

                                3. Tax progressivity slope for income (`ξ`) ∈ [0.0, 2.0]
                                 → Determines how steeply income tax increases with income

                              ---
                                 **Your reasoning should consider**:
                                 1. Does increasing τ discourage labor participation or increase inequality?
                                 2. Could raising τ_a reduce capital formation or disproportionately affect middle-class savers?
                                 3. Is current inequality high enough to justify steeper tax slopes (ξ)?
                                 4. Are government revenues sufficient to support Gt_prob at its desired level?
                                 5. Will changes help stabilize GDP growth and improve household welfare?

                              ---

                               **Constraints**:
                               - All values must be floats.
                               - Parameter bounds strictly enforced.
                               - Max absolute change per parameter: ±0.1

                               ---

                               **Output format**:
                               Respond only with the decision in strict JSON:
                               ```json
                               {{
                                 "Gt_prob": ...,  // e.g., 0.3 or 0.35
                                 "tau": ...,     // e.g., 0.30
                                 "xi": ...       // e.g., 0.8
                               }}
                               ```'''
            return prompt
        except Exception:
            return "You need to check whether the prompt here is consistent with the global_obs set in get_obs() within env_core"

    def get_action(self, global_obs_tensor, private_obs_tensor, gov_action=None, env=None):
        """
        Get expert actions by finding the closest observations in real data.

        Args:
            global_obs_tensor (torch.Tensor): Global observation tensor (unused for household).
            private_obs_tensor (torch.Tensor): Tensor of shape (N, 4) for household observations [ASSET, EDUC, INCOME, AGE].
            gov_action (optional): Government action (unused in this implementation).

        Returns:
            np.ndarray: Array of shape (N, 3) containing expert actions [LF, consumption_p, invest_p].
        """
        if self.agent_name == "government":
            if self.envs.government.type == "central_bank":
                private_gov_obs = global_obs_tensor[-2:].cpu().numpy()
                prompt = self.build_central_bank_prompt(global_obs_tensor.cpu().numpy(), private_gov_obs)
            elif self.envs.government.type == "pension":
                private_gov_obs = global_obs_tensor[-3:].cpu().numpy()
                prompt = self.build_pension_prompt(global_obs_tensor.cpu().numpy(), private_gov_obs)
            else:
                prompt = self.build_prompt(global_obs_tensor.cpu().numpy())

        elif self.agent_name == "tax_gov":
            prompt = self.build_prompt(global_obs_tensor.cpu().numpy())

        elif self.agent_name == "pension_gov":
            private_gov_obs = global_obs_tensor[-3:].cpu().numpy()
            prompt = self.build_pension_prompt(global_obs_tensor.cpu().numpy(), private_gov_obs)

        elif self.agent_name == "central_bank_gov":
            if hasattr(self.envs, "pension_gov"):
                private_gov_obs = global_obs_tensor[-5:-3].cpu().numpy()
            else:
                private_gov_obs = global_obs_tensor[-2:].cpu().numpy()

            prompt = self.build_central_bank_prompt(global_obs_tensor.cpu().numpy(), private_gov_obs)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # 或者 "deepseek-chat"
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            print("LLM Get Response Successfully")

            gpt_output = response.choices[0].message.content
            decision = self.extract_json(gpt_output)

            if decision is None:
                raise ValueError("Failed to parse GPT output.")

            if self.agent_name == "government":
                if self.envs.government.type == "central_bank":
                    base_interest_rate = float(decision["base_interest_rate"])
                    reserve_ratio = float(decision["reserve_ratio"])
                    return np.array([base_interest_rate, reserve_ratio])

                elif self.envs.government.type == "pension":
                    retire_age = float(decision["retire_age"])
                    contribution_rate = float(decision["contribution_rate"])
                    return np.array([retire_age, contribution_rate])

                else:
                    Gt_prob = float(decision["Gt_prob"])
                    tau = float(decision["tau"])
                    xi = float(decision["xi"])
                    return np.array([tau, xi, 0, 0, Gt_prob])

            elif self.agent_name == "tax_gov":
                Gt_prob = float(decision["Gt_prob"])
                tau = float(decision["tau"])
                xi = float(decision["xi"])
                return np.array([tau, xi, 0, 0, Gt_prob])

            elif self.agent_name == "pension_gov":
                retire_age = float(decision["retire_age"])
                contribution_rate = float(decision["contribution_rate"])
                return np.array([retire_age, contribution_rate])

            elif self.agent_name == "central_bank_gov":
                base_interest_rate = float(decision["base_interest_rate"])
                reserve_ratio = float(decision["reserve_ratio"])
                return np.array([base_interest_rate, reserve_ratio])


        except Exception as e:
            print("Error calling GPT or parsing result:", e)
            if self.agent_name == "government":
                if self.envs.government.type == "central_bank":
                    return np.array(global_obs_tensor[-2:])
                elif self.envs.government.type == "pension":
                    return np.array(global_obs_tensor[-2:])
                else:
                    return np.array([global_obs_tensor[9]], [global_obs_tensor[10]], 0, 0, [global_obs_tensor[11]])

            elif self.agent_name == "tax_gov":
                return np.array([global_obs_tensor[9]], [global_obs_tensor[10]], 0, 0, [global_obs_tensor[11]])

            elif self.agent_name == "pension_gov":
                return np.array(global_obs_tensor[-2:])

            elif self.agent_name == "central_bank_gov":
                if hasattr(self.envs, "pension_gov"):
                    return np.array(global_obs_tensor[-5:-4])
                else:
                    return np.array(global_obs_tensor[-2:])

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

        Args:
            gpt_output (str): GPT 返回的原始字符串

        Returns:
            Any: 解析得到的 JSON 数据（dict 或 list 等），若均失败则返回 None
        """

        # 1. 尝试整体解析
        try:
            return json.loads(gpt_output)
        except Exception:
            pass

        # 2. 尝试提取 ```json ...``` 代码块
        pattern_json_block = r"```json\s*([\s\S]*?)\s*```"
        match_json_block = re.search(pattern_json_block, gpt_output, re.IGNORECASE)
        if match_json_block:
            json_str = match_json_block.group(1).strip()
            try:
                return json.loads(json_str)
            except Exception:
                pass

        # 3. 尝试提取任意的 ``` ... ``` 代码块
        pattern_code_block = r"```([\s\S]*?)```"
        match_code_block = re.search(pattern_code_block, gpt_output)
        if match_code_block:
            json_str = match_code_block.group(1).strip()
            try:
                return json.loads(json_str)
            except Exception:
                pass

        # 4. 从文本中找到首个 '{' 和最后一个 '}'，提取中间内容尝试解析
        start_index = gpt_output.find("{")
        end_index = gpt_output.rfind("}")
        if start_index != -1 and end_index != -1 and start_index < end_index:
            possible_json_str = gpt_output[start_index:end_index + 1].strip()
            try:
                return json.loads(possible_json_str)
            except Exception:
                pass

        # 如果以上所有方式都失败，则返回 None
        return None
