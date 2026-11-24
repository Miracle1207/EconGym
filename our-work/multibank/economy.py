"""Nuestra economía: el EconomicSociety del original pero con nuestras 5 partes.

El original crea sus partes a mano dentro de EconomicSociety.__init__
(EconGym-original/env/env_core.py:20-33). Para no tocar su código, mientras se construye
cambiamos temporalmente las clases que usa por las nuestras (multibank/core/*).
"""

import contextlib

import env.env_core as env_core
from env.env_core import EconomicSociety

from multibank.core.business import Business
from multibank.core.central_bank import CentralBank
from multibank.core.commercial_bank import CommercialBank
from multibank.core.goverment import Government
from multibank.core.houses import Houses


def crear_gobierno(entity_args):
    """El original usa la misma clase para Hacienda, pensiones y banco central."""
    if entity_args.params.type == "central_bank":
        return CentralBank(entity_args)
    return Government(entity_args)


NUESTRAS_PARTES = {
    "Household": Houses,
    "Market": Business,
    "Bank": CommercialBank,
    "Government": crear_gobierno,
}


@contextlib.contextmanager
def usar_nuestras_partes():
    originales = {nombre: getattr(env_core, nombre) for nombre in NUESTRAS_PARTES}
    for nombre, clase in NUESTRAS_PARTES.items():
        setattr(env_core, nombre, clase)
    try:
        yield
    finally:
        for nombre, clase in originales.items():
            setattr(env_core, nombre, clase)


class MultiBankEconomy(EconomicSociety):
    def __init__(self, cfg):
        with usar_nuestras_partes():
            super().__init__(cfg)

    def reset(self, **custom_cfg):
        # El original no borra la inflación del episodio anterior, y la regla de Taylor la usa
        # en el primer año: sin esto, la misma semilla no da el mismo episodio.
        if hasattr(self, "inflation_rate"):
            del self.inflation_rate
        return super().reset(**custom_cfg)
