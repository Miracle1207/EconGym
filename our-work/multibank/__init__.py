"""MultiBank: nuestro código encima de EconGym-original.

Al importar multibank, EconGym-original entra en el path para que
`from entities.bank import Bank` (y el resto del original) funcione.
"""

import sys
from pathlib import Path

ORIGINAL = Path(__file__).resolve().parents[2] / "EconGym-original"

if str(ORIGINAL) not in sys.path:
    sys.path.insert(0, str(ORIGINAL))
