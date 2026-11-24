"""El entorno de RL: N bancos que aprenden, dentro de la economía de EconGym.

Dos formas de usarlo:

    # N bancos (para IPPO/MAPPO, experimentos E2, E6, E7...)
    env = MultiBankEnv(Config.pruebas(n_bancos=3))
    obs, info = env.reset(seed=1)                    # obs: (N, n_obs)
    obs, recompensas, terminado, truncado, info = env.step(acciones)   # acciones: (N, 5) en [-1, 1]

    # 1 banco con la API de Gymnasium (para SB3, experimento E1)
    env = BancoUnicoEnv(Config.pruebas())

Qué decide cada banco (5 números en [-1, 1], se escalan a):
    1. tipo de depósito       en [base - 1%, base]
    2. margen a empresas      en [base + margen_min, base + margen_max]
    3. margen a hogares       en [base + margen_min, base + margen_max]
    4. extra por riesgo       en [0, prima_max]   (por cada "vez" que el cliente va más endeudado que el típico)
    5. umbral de hogares      en [0, umbral_max]  (deuda/ingreso máxima que acepta)

Cada cliente acaba con su propio tipo:
    tipo = base + margen de su clase + extra × (riesgo - 1),  como mucho base + tipo_max_sobre_base

El resto de la economía va sola: hogares con su red BC, empresa con reglas y banco central
con la regla de Taylor (EconGym-original/agents/rule_based/government.py:13).
"""

import gymnasium as gym
import numpy as np
import torch

from multibank.banking import SistemaBancario, Tipos
from multibank.econgym import cargar_econgym
from settings.config import Config

N_ACCIONES = 5
NOMBRES_ACCIONES = ["deposito", "margen_empresas", "margen_hogares", "prima", "umbral"]

NOMBRES_OBS = [
    "tipo_base", "inflacion", "ratio_capital", "capital_vs_inicial",
    "cuota_empresas", "cuota_hogares", "cuota_depositos",
    "perdidas_ultimo_paso", "morosos_en_lista",
    "mi_margen_empresas", "mi_margen_hogares", "mi_prima", "mi_umbral",
    "rivales_margen_empresas", "rivales_margen_hogares", "rivales_prima",
    "progreso", "rentabilidad_empresas",
]


class MultiBankEnv:

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.eg = cargar_econgym(
            "optimal_monetary", n_hogares=c.n_hogares, episode_length=c.episode_length, semilla=c.seed,
            house_alg="bc", gov_alg="rule_based", firm_alg="rule_based", bank_alg="rule_based",
        )
        self.econ = self.eg.env
        self.N = c.n_bancos
        self.n_obs = len(NOMBRES_OBS)
        self.rng = np.random.default_rng(c.seed)
        self.sistema = None

    # ------------------------------------------------------------------ acciones
    def escalar(self, acciones, base):
        c = self.cfg
        u = (np.clip(np.asarray(acciones, float).reshape(self.N, N_ACCIONES), -1, 1) + 1) / 2
        return Tipos(
            deposito=np.maximum(base - 0.01 + 0.01 * u[:, 0], 0.0),
            empresas=base + c.margen_min + (c.margen_max - c.margen_min) * u[:, 1],
            hogares=base + c.margen_min + (c.margen_max - c.margen_min) * u[:, 2],
            prima=c.prima_max * u[:, 3],
            umbral=c.umbral_max * u[:, 4],
            techo=base + c.tipo_max_sobre_base,
        )

    def accion_desde(self, deposito_bajo_base, margen_empresas, margen_hogares, prima, umbral):
        """Al revés: de valores reales (en las mismas unidades) a una acción en [-1, 1]. Útil para reglas."""
        c = self.cfg
        u = np.array([
            1 - deposito_bajo_base / 0.01,
            (margen_empresas - c.margen_min) / (c.margen_max - c.margen_min),
            (margen_hogares - c.margen_min) / (c.margen_max - c.margen_min),
            prima / c.prima_max,
            umbral / c.umbral_max,
        ])
        return np.clip(2 * u - 1, -1, 1)

    # ------------------------------------------------------------------ API
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        semilla = int(self.rng.integers(2**31))
        np.random.seed(semilla)                             # la economía del original usa np.random
        torch.manual_seed(semilla)                          # y la red de los hogares, torch
        self.eg.reset()
        self.sistema = SistemaBancario(self.cfg, self.rng)
        self.sistema.iniciar(self.econ)
        base = float(self.econ.main_gov.base_interest_rate)
        self.tipos = self.escalar(np.zeros((self.N, N_ACCIONES)), base)
        self.sistema.tipos = self.tipos
        self.t = 0
        self.ya_quebrado = np.zeros(self.N, bool)
        return self.observar(), {}

    def step(self, acciones):
        cfg, econ, sis = self.cfg, self.econ, self.sistema
        base = float(econ.main_gov.base_interest_rate)
        self.tipos = self.escalar(acciones, base)
        vivo = ~sis.quebrado
        # Un banco quebrado ya no hace nada: no acepta a nadie y paga lo mínimo
        self.tipos.umbral[~vivo] = 0.0
        self.tipos.empresas[~vivo] = base + cfg.margen_max
        self.tipos.hogares[~vivo] = base + cfg.margen_max
        sis.tipos = self.tipos

        # La economía del original usa los tipos medios de nuestros bancos
        econ.bank.fijar_tipos(*sis.tipo_medio())
        _, _, econ_termina = self.eg.step()

        info_banca = sis.paso(econ, self.tipos)

        # Si a las empresas les falta crédito, invierten menos
        K_next = float(np.sum(econ.market.Kt_next))
        ap = cfg.apalancamiento_empresas
        K_fin = (1 - ap) * K_next + min(ap * K_next, info_banca["credito_empresas"])
        econ.market.Kt_next = econ.market.Kt_next * (K_fin / max(K_next, 1e-8))

        self.t += 1
        recompensas = self.recompensar(info_banca, vivo)
        terminado = sis.quebrado.copy()
        truncado = bool(econ_termina) or self.t >= cfg.episode_length or bool(terminado.all())
        info = {**info_banca, "economia_termina": bool(econ_termina)}
        return self.observar(), recompensas, terminado, truncado, info

    # ------------------------------------------------------------------ recompensa
    def recompensar(self, info, vivo_antes):
        cfg, sis = self.cfg, self.sistema
        if cfg.recompensa == "sigmoide":
            x = info["beneficio"]
            r = 1 / (1 + np.exp(-0.15 * np.sign(x) * np.log1p(np.abs(x))))
        else:
            r = info["beneficio"] / sis.K0
        r = np.where(vivo_antes, r, 0.0)
        r = r - cfg.lam_quiebra * info["nuevas_quiebras"]
        return r.astype(np.float32)

    # ------------------------------------------------------------------ observación
    def observar(self):
        cfg, sis, econ = self.cfg, self.sistema, self.econ
        base = float(econ.main_gov.base_interest_rate)
        cuota_e, cuota_h = sis.cuota_prestamos()
        cuota_d = sis.Dep / max(sis.Dep.sum(), 1e-12)
        Lb = sis.L.sum(axis=1)
        perdidas = sis.ultimo.get("perdida", np.zeros(self.N)) / np.maximum(Lb, 1e-12)
        morosos = sis.es_moroso().sum(axis=1) / sis.C
        m_e, m_h = self.tipos.empresas - base, self.tipos.hogares - base
        vivo = ~sis.quebrado
        prima = self.tipos.prima
        riv_e, riv_h, riv_p = np.zeros(self.N), np.zeros(self.N), np.zeros(self.N)
        if cfg.ver_rivales and self.N > 1:
            for b in range(self.N):
                otros = vivo & (np.arange(self.N) != b)
                if otros.any():
                    riv_e[b], riv_h[b], riv_p[b] = m_e[otros].mean(), m_h[otros].mean(), prima[otros].mean()
        K_total = max(float(np.sum(econ.market.Kt)), 1e-8)
        m = econ.market
        excedente = float(np.sum(m.price * m.Yt_j - m.WageRate * m.firm_labor_j)) if hasattr(m, "Yt_j") else 0.0
        obs = np.stack([
            np.full(self.N, base), np.full(self.N, float(np.clip(econ.inflation_rate, -1, 1)) if hasattr(econ, "inflation_rate") else 0.0),
            sis.ratio_capital(), sis.K / sis.K0,
            cuota_e, cuota_h, cuota_d,
            perdidas, morosos,
            m_e, m_h, prima, self.tipos.umbral / cfg.umbral_max,
            riv_e, riv_h, riv_p,
            np.full(self.N, self.t / cfg.episode_length), np.full(self.N, excedente / K_total),
        ], axis=1)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


class BancoUnicoEnv(gym.Env):
    """Un solo banco con la API de Gymnasium, para SB3 (comprobación y E1)."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config()
        cfg.n_bancos = 1
        self.multi = MultiBankEnv(cfg)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.multi.n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (N_ACCIONES,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.multi.reset(seed=seed)
        return obs[0], info

    def step(self, accion):
        obs, r, terminado, truncado, info = self.multi.step(np.asarray(accion)[None, :])
        return obs[0], float(r[0]), bool(terminado[0]), bool(truncado and not terminado[0]), info
