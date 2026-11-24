"""Gobierno (Hacienda y pensiones). Lo dejamos como en el original.

Original: EconGym-original/entities/government.py (class Government)
- tax_step()     -> cobra impuestos (tau=0.263 por defecto), gasta Gt_prob*Y y
                    se endeuda: Bt_next = (1+tipo_base)*Bt + gasto - impuestos (government.py:97).
- pension_step() -> solo con hogares OLG.
Esa deuda pública Bt la financia el banco antes que a la empresa (bank.py).

No pensamos tocarlo.
"""

from entities.government import Government as OriginalGovernment


class Government(OriginalGovernment):
    pass
