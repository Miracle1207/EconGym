"""Tests del entorno completo (economía pequeña para que vayan rápido)."""

import numpy as np
import pytest

from multibank.env import N_ACCIONES as A, BancoUnicoEnv, MultiBankEnv
from settings.config import Config


@pytest.fixture(scope="module")
def env3():
    return MultiBankEnv(Config.pruebas(n_bancos=3, episode_length=15))


def test_forma_de_obs_y_recompensas(env3):
    obs, _ = env3.reset(seed=1)
    assert obs.shape == (3, env3.n_obs)
    obs, r, term, trunc, info = env3.step(np.zeros((3, A)))
    assert obs.shape == (3, env3.n_obs) and r.shape == (3,) and term.shape == (3,)
    assert np.isfinite(obs).all() and np.isfinite(r).all()


def test_el_balance_cuadra_en_cada_paso_con_acciones_al_azar(env3):
    # comprobar_balance() lanza un error si activo != depósitos + capital en algún banco
    rng = np.random.default_rng(0)
    for seed in (1, 2):
        env3.reset(seed=seed)
        for _ in range(15):
            _, _, _, trunc, _ = env3.step(rng.uniform(-1, 1, (3, A)))
            env3.sistema.comprobar_balance()
            if trunc:
                break


def test_misma_semilla_mismo_resultado(env3):
    def episodio():
        env3.reset(seed=7)
        return [env3.step(np.zeros((3, A)))[1] for _ in range(5)]
    a, b = episodio(), episodio()
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_un_banco_quebrado_no_presta_y_sigue_terminado(env3):
    env3.reset(seed=1)
    s = env3.sistema                               # forzamos la quiebra del banco 0
    perdida = 3 * s.K[0]                           # pierde tres veces su capital
    s.K[0] -= perdida
    s.Liq[0] -= perdida
    _, r, term, _, info = env3.step(np.zeros((3, A)))
    assert term[0] and info["nuevas_quiebras"][0]
    esperado = info["beneficio"][0] / s.K0[0] - env3.cfg.lam_quiebra
    assert np.isclose(r[0], esperado, atol=1e-5)  # se lleva la penalización
    prestado = env3.sistema.L[0].sum()
    _, r, term, _, _ = env3.step(np.zeros((3, A)))
    assert term[0] and r[0] == 0.0                 # después ya no recibe nada
    assert env3.sistema.L[0].sum() <= prestado     # y no presta más


def test_a_un_moroso_no_se_le_renueva(env3):
    env3.reset(seed=1)
    s = env3.sistema
    hogar = s.F + int(np.argmax(s.objetivo[s.F:]))
    s.ultimo_impago[:, hogar] = s.t + 1            # acaba de no pagar en todos los bancos
    antes = s.L[:, hogar].sum()
    env3.step(np.zeros((3, A)))
    assert s.L[:, hogar].sum() <= antes            # nadie le presta más


def test_banco_unico_cumple_la_api_de_gymnasium():
    from stable_baselines3.common.env_checker import check_env
    check_env(BancoUnicoEnv(Config.pruebas(episode_length=5)), warn=False)


def test_cada_cliente_paga_segun_su_riesgo(env3):
    env3.reset(seed=1)
    env3.step(np.zeros((3, A)))
    s = env3.sistema
    a = np.zeros((3, A))
    a[:, 3] = 1.0                                  # extra por riesgo al máximo
    tipos = env3.escalar(a, 0.02)
    r = s.tipo_por_cliente(tipos)[0, s.F:]
    riesgo = s.riesgo[s.F:]
    seguro, arriesgado = np.argmin(riesgo), np.argmax(riesgo)
    assert r[arriesgado] > r[seguro]               # al más endeudado le cobra más
    assert r.max() <= tipos.techo + 1e-12          # pero nunca por encima del techo
