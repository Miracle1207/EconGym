"""Empresa. De momento es la del original, sin cambios.

Original: EconGym-original/entities/market.py (class Market)
- step()  -> produce Yt = Z * K^alpha * L^(1-alpha). Su capital Kt lo financia el banco
             (Bank.compute_next_kt): la empresa pide todo su capital prestado.
- Política: regla fija (rule_based), competencia perfecta, firm_n = 1.

Qué tocaremos aquí (fases 2-3):
- La empresa puede hacer impago (probabilidad que sube con el tipo de interés).
- Su demanda de crédito se reparte entre los N bancos con un softmax sobre los tipos.
"""

from entities.market import Market


class Business(Market):
    pass
