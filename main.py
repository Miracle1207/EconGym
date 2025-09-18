from env.env_core import EconomicSociety
from agents.rule_based.rules_core import rule_agent
from agents.rl.ppo_agent import ppo_agent
from agents.rl.sac_agent import sac_agent
from agents.behavior_cloning.bc_agent import bc_agent
from agents.rl.ddpg_agent import ddpg_agent
from agents.real_data.real_data import real_agent
from agents.llm.llm_agent import llm_agent
from agents.data_based_agent import data_agent
from utils.seeds import set_seeds
from utils.config import load_config
import os
import argparse
from runner import Runner

agent_algorithms = {
    "real": real_agent,
    "ppo": ppo_agent,
    "sac": sac_agent,
    "rule_based": rule_agent,
    "bc": bc_agent,
    "llm": llm_agent,
    "data_based": data_agent,
    "ddpg": ddpg_agent,
    "saez": rule_agent,
    "us_federal": rule_agent,
}


def select_agent(alg, agent_name, agent_type, env, trainer_config):
    if alg not in agent_algorithms:
        raise ValueError(f"Unsupported algorithm: {alg}")
    if alg == "saez" or alg == "us_federal":
        env.government['tax'].tax_type = alg
    return agent_algorithms[alg](env, trainer_config, agent_name=agent_name, type=agent_type)


def setup_government_agents(config, env):
    """
    Initialize multiple government agents based on config if problem_scene is multi_gov.
    """
    if isinstance(env.government, dict):
        gov_algs = {}

        for gov_type, gov_agent in env.government.items():
            alg_name = gov_type + "_gov_alg"
            gov_key = alg_name if alg_name in config['Trainer'] else 'gov_alg'

            # Try to get the appropriate agent config, and select agent if found
            gov_alg = None
            if gov_key in config['Trainer']:
                gov_alg = select_agent(config['Trainer'].get(gov_key), "government", gov_type, env, config['Trainer'])

            if gov_alg is None:  # Log a warning if no algorithm is found for a given government type
                print(f"Warning: No algorithm found for government type '{gov_type}' using key '{gov_key}'")

            gov_algs[gov_type] = gov_alg

        return gov_algs

    else:
        raise ValueError("Government should be a dict")


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_scene", type=str, default='inflation_control', help="Problem scene to simulate")
    args = parser.parse_args()

    config = load_config(args.problem_scene)
    set_seeds(config['Trainer']['seed'], cuda=config['Trainer']['cuda'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device_num'])

    env = EconomicSociety(config['Environment'])

    # Equip agent algorithm for economic roles
    house_agent = select_agent(config['Trainer']['house_alg'], "households", env.households.type, env,
                               config['Trainer'])
    firm_agent = select_agent(config['Trainer']['firm_alg'], "market", env.market.type, env, config['Trainer'])
    bank_agent = select_agent(config['Trainer']['bank_alg'], "bank", env.bank.type, env, config['Trainer'])

    gov_agents = setup_government_agents(config, env)

    print(f"Problem Scene: {env.problem_scene}")
    print(f"Households_n: {env.households.households_n}")

    test_mode = config.get('Trainer', {}).get('test', False)
    runner = Runner(
        env,
        config['Trainer'],
        house_agent=house_agent,
        government_agent=gov_agents,
        firm_agent=firm_agent,
        bank_agent=bank_agent,
    )
    if test_mode:
        runner.test()
    else:
        runner.run()
