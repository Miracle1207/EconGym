# Nuestro trabajo

Aquí va todo lo que hemos hecho nosotros (código, experimentos, resultados, informe).
El código original de EconGym está en `../EconGym-original/` y no se toca.

## Qué hay

| Archivo | Qué es |
|---|---|
| `settings/config.py` | Todos los parámetros (`Config()` = experimentos, `Config.pruebas()` = economía pequeña para probar) |
| `multibank/env.py` | El entorno de RL: `MultiBankEnv` (N bancos) y `BancoUnicoEnv` (1 banco, API de Gymnasium para SB3) |
| `multibank/banking.py` | Los N bancos: préstamos, intereses, impagos, lista de morosos, capital mínimo y quiebra |
| `multibank/reglas.py` | Reglas fijas para comparar con el RL: prudente, agresiva, clásica y competitiva |
| `multibank/core/` | Las 5 partes de la economía, heredando del original |
| `multibank/econgym.py` | El puente que carga la economía del original |
| `multibank/visual.py` + `visor.html` | Graba una simulación y genera el visor |
| `tests/` | Tests (`uv run pytest` desde la raíz) |

## Cómo se usa

Desde `our-work/`:

```bash
# Ver una simulación (abre el navegador)
uv run python -m multibank.visual --abrir
uv run python -m multibank.visual --politicas agresiva,clasica,prudente --hogares 1000 --abrir

# Tests (desde la raíz del repo)
uv run pytest
```

En código:

```python
from multibank.env import MultiBankEnv, BancoUnicoEnv
from settings.config import Config

env = MultiBankEnv(Config(n_bancos=3))          # 1000 hogares, 50 empresas
obs, info = env.reset(seed=1)                   # obs: (3, 18)
obs, recompensas, terminado, truncado, info = env.step(acciones)   # acciones: (3, 5) en [-1, 1]

# Un banco con SB3
from stable_baselines3 import PPO
PPO("MlpPolicy", BancoUnicoEnv(Config.pruebas())).learn(10_000)
```

## Qué decide cada banco (acción: 5 números en [-1, 1])

1. Tipo de depósito: entre el tipo base − 1% y el tipo base.
2. Margen a empresas: entre base + 1% y base + 8%.
3. Margen a hogares: entre base + 1% y base + 8%.
4. Extra por riesgo: de 0 a 5% por cada "vez" que el cliente va más endeudado que el típico de su clase.
5. Umbral: la deuda máxima, en años de ingreso (0 a 4), de un hogar al que acepta prestar.

Cada cliente paga su propio tipo: base + margen de su clase + extra × (riesgo − 1), como mucho base + 15%.

## Qué ve cada banco (observación: 18 números)

Tipo base, inflación, su capital frente a sus préstamos, su capital frente al inicial, sus cuotas
(empresas, hogares y depósitos), sus pérdidas del último año, sus morosos, sus últimas decisiones
(márgenes, extra por riesgo, umbral), los márgenes y el extra por riesgo de los rivales (si `ver_rivales`), el año del episodio y la rentabilidad de las empresas.

## Recompensa

Cambio de su capital en el año / capital inicial (con signo). Si quiebra: − λ (`lam_quiebra`) y el episodio
termina para ese banco (`terminated`). El episodio acaba a los 30 años (`truncated`).
