"""Puente con el EconGym original: lo carga y lo deja listo para usar desde our-work.

El código original usa rutas relativas (cfg/..., agents/models/...), así que hay que
ejecutarlo "desde dentro" de EconGym-original. Este módulo se encarga de eso.

Uso:
    from multibank.econgym import cargar_econgym
    eg = cargar_econgym("optimal_monetary")
    obs = eg.reset()
    obs, recompensas, done = eg.step()
"""

import contextlib
import os

from multibank import ORIGINAL


@contextlib.contextmanager
def dentro_del_original():
    """Entra temporalmente en EconGym-original (para sus rutas relativas)."""
    antes = os.getcwd()
    os.chdir(ORIGINAL)
    try:
        yield
    finally:
        os.chdir(antes)


class EconGym:
    """Nuestra economía (multibank/parts) + los agentes de cada rol, como en su main.py."""

    def __init__(self, problem_scene="optimal_monetary", overrides=None, n_hogares=None,
                 n_empresas=None, episode_length=None, semilla=None):
        with dentro_del_original():
            import main as original_main
            from multibank.economy import MultiBankEconomy
            from runner import Runner
            from utils.config import load_config
            from utils.seeds import set_seeds

            config = load_config(problem_scene)
            config.Trainer.wandb = False            # sin swanlab
            for e in config.Environment.Entities:
                if e.entity_name == "households" and n_hogares is not None:
                    e.entity_args.params.households_n = n_hogares
                if e.entity_name == "market" and n_empresas is not None and n_empresas > 1:
                    # Varias empresas: el mercado de competencia monopolística del original
                    e.entity_args.params.type = "monopolistic_competition"
                    e.entity_args.monopolistic_competition.firm_n = n_empresas
            if episode_length is not None:
                config.Environment.env_core.episode_length = episode_length
            for clave, valor in (overrides or {}).items():
                config.Trainer[clave] = valor
            if semilla is not None:
                config.Trainer.seed = semilla
            set_seeds(config.Trainer.seed, cuda=config.Trainer.cuda)

            env = MultiBankEconomy(config.Environment)
            t = config.Trainer
            agentes = dict(
                house_agent=original_main.select_agent(t.house_alg, "households", env.households.type, env, t),
                firm_agent=original_main.select_agent(t.firm_alg, "market", env.market.type, env, t),
                bank_agent=original_main.select_agent(t.bank_alg, "bank", env.bank.type, env, t),
                government_agent=original_main.setup_government_agents(config, env),
            )
            # Usamos su Runner solo por sus utilidades para pedir acciones a los agentes
            self.runner = Runner(env, t, **agentes)

        self.config = config
        self.env = self.runner.envs
        self.t = 0
        self.obs = None

    def reset(self):
        self.t = 0
        self.obs = self.env.reset()
        return self.obs

    def step(self):
        """Un paso: todos los agentes eligen acción con su política y el entorno avanza."""
        obs_tensor = self.runner._get_tensor_inputs(self.obs)
        _, acciones = self.runner.agents_get_action(obs_tensor)
        self.obs, recompensas, done = self.env.step(acciones, self.t)
        self.t += 1
        return self.obs, recompensas, done


def cargar_econgym(problem_scene="optimal_monetary", n_hogares=None, n_empresas=None,
                   episode_length=None, semilla=None, **overrides):
    return EconGym(problem_scene, overrides, n_hogares=n_hogares, n_empresas=n_empresas,
                   episode_length=episode_length, semilla=semilla)


if __name__ == "__main__":
    eg = cargar_econgym("optimal_monetary")
    eg.reset()
    banco = eg.env.bank
    print(f"Arrancado: {eg.config.Environment.env_core.problem_scene}, "
          f"{eg.env.households.households_n} hogares")
    for paso in range(10):
        _, recompensas, done = eg.step()
        print(f"paso {paso}: tipo préstamo={banco.lending_rate:.4f} "
              f"tipo depósito={banco.deposit_rate:.4f} "
              f"recompensa banco={recompensas['bank'].item():.3f} done={done}")
