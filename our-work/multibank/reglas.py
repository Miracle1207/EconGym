"""Reglas fijas para comparar con el RL (benchmarks de E1 y suelo competitivo de E6).

Cada regla es una función obs -> acción en [-1, 1] para UN banco, igual que una política de RL:

    regla = REGLAS["prudente"](env)
    accion = regla(obs_del_banco)

Valores en las unidades de verdad (márgenes sobre el tipo base, umbral en deuda/ingreso):
    prudente  : margen 6%, extra por riesgo 3%, umbral 0,8 (solo hogares que deben menos de ~10 meses de ingreso)
    agresiva  : margen 1%, sin extra por riesgo, umbral 4   (presta a casi todos, barato y a todos igual)
    clasica   : margen 2% + subida según las pérdidas recientes, extra por riesgo 1%, umbral 2
    competitiva (E6, suelo): baja el margen mientras gana dinero y lo sube si pierde
"""

import numpy as np

from multibank.env import NOMBRES_OBS

I = {n: i for i, n in enumerate(NOMBRES_OBS)}


def prudente(env):
    a = env.accion_desde(0.01, 0.06, 0.06, 0.03, 0.8)
    return lambda obs: a


def agresiva(env):
    a = env.accion_desde(0.0, 0.01, 0.01, 0.0, 4.0)
    return lambda obs: a


def clasica(env, prima_por_perdida=5.0):
    def regla(obs):
        perdidas = float(obs[I["perdidas_ultimo_paso"]])
        margen = float(np.clip(0.02 + prima_por_perdida * perdidas, env.cfg.margen_min, env.cfg.margen_max))
        return env.accion_desde(0.005, margen, margen, 0.01, 2.0)
    return regla


def competitiva(env, paso=0.002):
    estado = {"margen": 0.04, "capital": None}

    def regla(obs):
        capital = float(obs[I["capital_vs_inicial"]])
        if estado["capital"] is not None:
            gana = capital > estado["capital"]
            estado["margen"] += -paso if gana else paso
            estado["margen"] = float(np.clip(estado["margen"], env.cfg.margen_min, env.cfg.margen_max))
        estado["capital"] = capital
        return env.accion_desde(0.005, estado["margen"], estado["margen"], 0.01, 2.0)
    return regla


REGLAS = {"prudente": prudente, "agresiva": agresiva, "clasica": clasica, "competitiva": competitiva}


def jugar(env, politicas, seed=1):
    """Juega un episodio. `politicas`: una función obs->acción por banco. Devuelve la recompensa total por banco."""
    obs, _ = env.reset(seed=seed)
    total = np.zeros(env.N)
    while True:
        acciones = np.stack([pol(obs[b]) for b, pol in enumerate(politicas)])
        obs, r, terminado, truncado, info = env.step(acciones)
        total += r
        if truncado:
            return total
