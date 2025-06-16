from env.env_core import EconomicSociety
from agents.rule_based import rule_agent
from agents.ppo_agent import ppo_agent
from agents.bc_agent import bc_agent
from agents.ddpg_agent import ddpg_agent
from agents.real_data.real_data import real_agent
from agents.llm_agent import llm_agent
from agents.data_based_agent import data_agent
from utils.seeds import set_seeds
from utils.config import load_config
import os
import argparse
from runner import Runner
from entities.central_bank_gov import CentralBankGovernment
from entities.pension_gov import PensionGovernment


def select_agent(alg, agent_name, env, trainer_config):
    agent_constructors = {
        "real": real_agent,
        "ppo": ppo_agent,
        "rule_based": rule_agent,
        "bc": bc_agent,
        "llm": llm_agent,
        "data_based": data_agent,
        "ddpg": ddpg_agent,
        "saez": rule_agent,
        "us_federal": rule_agent,
    }

    if alg not in agent_constructors:
        raise ValueError(f"Unsupported algorithm: {alg}")
    if alg == "saez" or alg == "us_federal":
        env.government.tax_type = alg
    return agent_constructors[alg](env, trainer_config, agent_name=agent_name)


def setup_government_agents(config, env):
    """
    Initialize multiple government agents based on config if problem_scene is multi_gov.
    """
    if config.Environment.env_core.problem_scene not in {"tre_government", "dbl_government", "sgl_government"}:
        return {
            "government_agent": select_agent(config['Trainer']['gov_alg'], "government", env,
                                             config['Trainer']) if 'gov_alg' in config['Trainer'] else None,
            "central_bank_gov_agent": None,
            "tax_gov_agent": None,
            "pension_gov_agent": None
        }
    else:
        agents = {
            "government_agent": None,
            "central_bank_gov_agent": None,
            "tax_gov_agent": None,
            "pension_gov_agent": None
        }
        if 'central_bank_gov_alg' in config['Trainer']:
            agents["central_bank_gov_agent"] = select_agent(config['Trainer']['central_bank_gov_alg'],
                                                            "central_bank_gov", env, config['Trainer'])

        if 'tax_gov_alg' in config['Trainer']:
            agents["tax_gov_agent"] = select_agent(config['Trainer']['tax_gov_alg'], "tax_gov", env, config['Trainer'])

        if 'pension_gov_alg' in config['Trainer']:
            agents["pension_gov_agent"] = select_agent(config['Trainer']['pension_gov_alg'], "pension_gov", env,
                                                       config['Trainer'])

        return agents


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_scene", type=str, default='inflation_control', help="Problem scene to simulate")
    args = parser.parse_args()

    config = load_config(args.problem_scene)
    set_seeds(config['Trainer']['seed'], cuda=config['Trainer']['cuda'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device_num'])

    if 'pension_gov_alg' not in config['Trainer']:
        env = EconomicSociety(config['Environment'], invalid_gov="pension_gov")
    elif 'central_bank_gov_alg' not in config['Trainer']:
        env = EconomicSociety(config['Environment'], invalid_gov="central_bank_gov")
    else:
        env = EconomicSociety(config['Environment'])

    house_agent = select_agent(config['Trainer']['house_alg'], "household", env, config['Trainer'])
    firm_agent = select_agent(config['Trainer']['firm_alg'], "market", env, config['Trainer'])
    bank_agent = select_agent(config['Trainer']['bank_alg'], "bank", env, config['Trainer'])

    gov_agents = setup_government_agents(config, env)

    print(f"Problem Scene: {env.problem_scene}")
    print(f"Households_n: {env.households.households_n}")

    test_mode = config.get('Trainer', {}).get('test', False)

    if test_mode:
        heter_house_agent = None
        if config['Trainer'].get('heterogeneous_house_agent', False):
            heter_house_agent = select_agent(config['Trainer']['heter_house_alg'], "household", env, config['Trainer'])

        runner = Runner(
            env,
            config['Trainer'],
            house_agent=house_agent,
            government_agent=gov_agents['government_agent'],
            firm_agent=firm_agent,
            bank_agent=bank_agent,
            heter_house=heter_house_agent,
            central_bank_gov_agent=gov_agents['central_bank_gov_agent'],
            tax_gov_agent=gov_agents['tax_gov_agent'],
            pension_gov_agent=gov_agents['pension_gov_agent']
        )
        runner.test()
    else:
        runner = Runner(
            env,
            config['Trainer'],
            house_agent=house_agent,
            government_agent=gov_agents['government_agent'],
            firm_agent=firm_agent,
            bank_agent=bank_agent,
            central_bank_gov_agent=gov_agents['central_bank_gov_agent'],
            tax_gov_agent=gov_agents['tax_gov_agent'],
            pension_gov_agent=gov_agents['pension_gov_agent']
        )
        runner.run()
