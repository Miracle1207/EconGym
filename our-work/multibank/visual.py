"""Graba una simulación del entorno MultiBank y genera el visor: our-work/viz/simulacion.html

Uso (desde our-work):
    uv run python -m multibank.visual --abrir
    uv run python -m multibank.visual --politicas agresiva,clasica,prudente --hogares 1000 --abrir
    uv run python -m multibank.visual --politicas azar,azar --semilla 3

Cada banco juega con una política: una de las reglas (prudente, agresiva, clasica, competitiva)
o "azar". Cuando haya modelos entrenados, se podrán pasar aquí también.
"""

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np

from multibank.env import N_ACCIONES, MultiBankEnv
from multibank.reglas import REGLAS
from settings.config import Config

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI.parent / "viz" / "simulacion.html"
PLANTILLA = AQUI / "visor.html"


def r(x, d=0):
    return [round(float(v), d) for v in np.ravel(x)]


def foto(env, info, recompensas):
    s, e, t = env.sistema, env.econ, env.tipos
    F = s.F
    Lb = s.L.sum(axis=1)
    deuda_cliente = s.L.sum(axis=0)
    principal = np.where(deuda_cliente > 0, s.L.argmax(axis=0), -1)
    moroso = s.es_moroso().any(axis=0)
    ce, ch = s.cuota_prestamos()
    # El tipo que paga cada cliente a su banco principal (sin deuda: el más barato que le ofrecen)
    tipos_c = s.tipo_por_cliente(t)
    tipo_cliente = np.where(principal >= 0, tipos_c[np.maximum(principal, 0), np.arange(s.C)], tipos_c.min(axis=0))
    return {
        "base": float(info["base"]),
        "pib": float(e.main_gov.GDP),
        "inflacion": float(getattr(e, "inflation_rate", 0.0)),
        "bancos": [{
            "dep": float(t.deposito[b]), "emp": float(t.empresas[b]), "hog": float(t.hogares[b]),
            "prima": float(t.prima[b]), "umbral": float(t.umbral[b]),
            "capital": float(s.K[b]), "capital0": float(s.K0[b]), "ratio": float(s.K[b] / max(Lb[b], 1e-9)),
            "p_emp": float(s.L[b, :F].sum()), "p_hog": float(s.L[b, F:].sum()), "bonos": float(s.G[b]),
            "depositos": float(s.Dep[b]), "liquidez": float(s.Liq[b]),
            "beneficio": float(info["beneficio"][b]), "perdida": float(info["perdida"][b]),
            "recompensa": float(recompensas[b]), "quebrado": bool(s.quebrado[b]),
            "cuota_emp": float(ce[b]), "cuota_hog": float(ch[b]), "cuota_dep": float(s.Dep[b] / max(s.Dep.sum(), 1e-9)),
            "morosos": int(s.es_moroso()[b].sum()),
        } for b in range(s.N)],
        "empresas": {
            "deuda": r(deuda_cliente[:F]), "banco": [int(x) for x in principal[:F]],
            "impaga": [int(x) for x in info["impaga"][:F]], "p": r(info["p_impago"][:F], 4),
            "moroso": [int(x) for x in moroso[:F]], "racionado": r(info["racionado"][:F]),
            "tipo": r(tipo_cliente[:F], 4),
        },
        "hogares": {
            "riqueza": r(e.households.at), "ingreso": r(e.households.income),
            "deuda": r(deuda_cliente[F:]), "banco": [int(x) for x in principal[F:]],
            "impaga": [int(x) for x in info["impaga"][F:]], "moroso": [int(x) for x in moroso[F:]],
            "tipo": r(tipo_cliente[F:], 4),
            "edad": [int(x) for x in np.ravel(e.households.age)],
        },
    }


def politica(nombre, env, rng):
    if nombre == "azar":
        return lambda obs: rng.uniform(-1, 1, N_ACCIONES)
    return REGLAS[nombre](env)


def grabar(cfg, nombres, semilla):
    env = MultiBankEnv(cfg)
    rng = np.random.default_rng(semilla)
    politicas = [politica(n, env, rng) for n in nombres]
    obs, _ = env.reset(seed=semilla)
    historia, motivo = [], "se completó el episodio"
    while True:
        acciones = np.stack([p(obs[b]) for b, p in enumerate(politicas)])
        obs, recompensas, terminado, truncado, info = env.step(acciones)
        historia.append(foto(env, info, recompensas))
        if truncado:
            if terminado.all():
                motivo = "quebraron todos los bancos"
            elif info["economia_termina"] and env.t < cfg.episode_length:
                motivo = "la economía del original se rompió"
            break
    return {
        "politicas": nombres, "n_empresas": env.sistema.F, "n_hogares": env.sistema.H,
        "capital_minimo": cfg.capital_minimo, "semilla": semilla, "fin": motivo,
        "empresa_peso": r(env.sistema.peso_empresa, 5),
        "historia": historia,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--politicas", default="agresiva,clasica,prudente",
                   help="una por banco: " + ", ".join(list(REGLAS) + ["azar"]))
    p.add_argument("--hogares", type=int, default=100, help="100 (5 empresas) o 1000 (50 empresas)")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--abrir", action="store_true", help="abrir el visor en el navegador al acabar")
    args = p.parse_args()

    nombres = [n.strip() for n in args.politicas.split(",")]
    cfg = Config(n_bancos=len(nombres)) if args.hogares >= 1000 else Config.pruebas(n_bancos=len(nombres))
    datos = grabar(cfg, nombres, args.semilla)
    html = PLANTILLA.read_text(encoding="utf-8").replace("__DATOS__", json.dumps(datos, separators=(",", ":")))
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(html, encoding="utf-8")
    print(f"{len(datos['historia'])} años grabados ({datos['fin']}) -> {SALIDA}")
    if args.abrir:
        webbrowser.open(SALIDA.as_uri())


if __name__ == "__main__":
    main()
