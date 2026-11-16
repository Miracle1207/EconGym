# EconGym-MultiBank

Trabajo de clase basado en **EconGym**: https://github.com/Miracle1207/EconGym
(paper: https://arxiv.org/abs/2506.12110).

- `EconGym-original/` → código original de EconGym, sin modificar.
- `our-work/` → lo que hemos hecho nosotros.

## Cómo ejecutarlo

Necesitas [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # crea .venv e instala dependencias (Python 3.11)
cd EconGym-original
uv run python main.py --problem_scene "optimal_monetary"
```
