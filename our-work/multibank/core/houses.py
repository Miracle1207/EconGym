"""Hogares: los del original + su deuda real.

Original: EconGym-original/entities/households.py (class Household)
- households_init() -> saca cada hogar al azar de la encuesta SCF 2022 (ponderando por WGT),
                       pero solo guarda sus activos (ASSET), no su deuda.
- step()            -> ramsey_step(): ahorran en el banco (at_next) a deposit_rate.
- is_terminal()     -> el episodio entero acaba si un hogar baja de at_min (-1e6).
Su política es la red BC congelada (ramsey_bc_net.pt), no la entrenamos.

Lo nuestro: households_init() guarda también, de cada hogar sacado:
- self.debt_init: su deuda real (DEBT). El sistema bancario la usa como préstamo inicial.
- self.liq_share: qué parte de sus activos tiene en el banco (LIQ / ASSET: cuentas corrientes
  y de ahorro). En los datos es ~5%; el resto son casas, acciones, coches...
"""

import numpy as np

from entities.households import Household


class Houses(Household):

    def get_real_data(self, age_limit=None):
        data = super().get_real_data(age_limit)
        # data = [ASSET, EDUC, INCOME, AGE, LF, consumo, inversión, WGT]; metemos DEBT antes de WGT
        import os
        import pandas as pd
        df = pd.read_csv(os.path.join(os.path.abspath('.'), "agents/data/advanced_scfp2022_1110.csv"))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if age_limit is not None:
            df = df[df['AGE'] == age_limit]
        return data[:-1] + [df['DEBT'].values, df['LIQ'].values] + data[-1:]

    def households_init(self):
        # Igual que el original, pero recogiendo también la deuda.
        data = self.get_real_data()
        self.real_e = data[1]
        (self.at_init, self.e_init, self.it_init, self.age_init, self.work_init,
         self.c_init, self.investment_init, self.debt_init, liq) = self.sample_real_data(data)
        self.liq_share = np.clip(liq / np.maximum(self.at_init, 1.0), 0.0, 1.0)
        self.generate_c_init(self.age_init)
        self.saving_init = 1 - self.c_init
        self.e_initial(self.households_n)
        self.initial_action = np.concatenate((self.saving_init, self.work_init * self.h_max), axis=1)
