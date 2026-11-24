"""Banco central. En el original es un Government con type='central_bank'.

Original: EconGym-original/entities/government.py (class Government)
- Su acción es (tipo base, coeficiente de reservas).
- Recompensa: get_reward_central() (inflación y crecimiento).
- Política habitual: PPO, o la regla de Taylor de
  EconGym-original/agents/rule_based/government.py:13 (cb_rule_taylor).
- El banco comercial recorta sus tipos alrededor del tipo base (bank.py:41-42):
  préstamo en [base+1%, base+3%], depósito en [base-1%, base].

Lo nuestro: en los experimentos lo fijamos con la regla de Taylor, y el tipo base se mueve
poco a poco (suavizado), como hacen los bancos centrales de verdad:
    tipo_t = suavizado * tipo_{t-1} + (1 - suavizado) * lo que pide la regla
Sin suavizar, con la inflación tan movida del original, el tipo saltaba de 0% a 10% de un año a otro.
"""

from entities.government import Government as OriginalGovernment


class CentralBank(OriginalGovernment):
    suavizado = 0.8

    def reset(self, **custom_cfg):
        # El reset del original no devuelve el tipo base a su valor inicial: lo hacemos aquí,
        # para que cada episodio empiece igual (y la misma semilla dé el mismo resultado).
        if not hasattr(self, "inicial"):
            self.inicial = (self.base_interest_rate, self.reserve_ratio)
        self.base_interest_rate, self.reserve_ratio = self.inicial
        super().reset(**custom_cfg)
        self.tipo_anterior = None

    def get_action(self, actions, firm_n):
        super().get_action(actions, firm_n)
        if getattr(self, "tipo_anterior", None) is not None:
            self.base_interest_rate = self.suavizado * self.tipo_anterior + (1 - self.suavizado) * float(self.base_interest_rate)
        self.tipo_anterior = float(self.base_interest_rate)
