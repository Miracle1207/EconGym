"""Parámetros del proyecto MultiBank (decisiones del plan, 23-11-2026).

Aquí van todos los números que se tocan entre experimentos.
Cada experimento crea un Config cambiando solo lo que necesita, por ejemplo:

    cfg = Config(n_bancos=3, lam_quiebra=2.0, seed=4)
    cfg = Config.pruebas()          # economía pequeña para programar rápido
"""

from dataclasses import dataclass, replace


@dataclass
class Config:
    # --- Economía (1 empresa por cada ~20 hogares, como en la realidad) ---
    n_hogares: int = 1000
    n_empresas: int = 50
    deuda_real: bool = True                 # cargar la deuda de cada hogar de la encuesta SCF 2022 (columna DEBT)

    # --- Bancos ---
    n_bancos: int = 2
    capital_minimo: float = 0.105           # capital >= 10,5% de lo prestado (Basilea III: 8% + 2,5%)
    capital_inicial: float = 0.15           # capital al empezar = 15% de lo que presta cada banco
    # Sin reservas y sin pesos por riesgo: todos los préstamos cuentan igual.

    # --- Clientes: empresas y hogares ---
    apalancamiento_empresas: float = 0.7    # las empresas financian el 70% de su capital con préstamos
    amortizacion_empresas: float = 0.2      # préstamos a 5 años: se devuelve un 20% al año
    amortizacion_hogares: float = 0.05      # hipotecas: se devuelve un 5% al año (20 años)
    persistencia_suerte: float = 0.8        # la suerte de cada empresa se parece a la del año anterior
    volatilidad_suerte: float = 0.4         # cuánto cambia la suerte de cada empresa cada año
    ingreso_minimo: float = 5000.0          # para no dividir entre 0 en deuda/ingreso

    # --- Probabilidad de impago: pmax * sigmoide(k * (carga - c0)) ---
    # empresas: carga = intereses / lo que gana la empresa; hogares: (intereses + amortización) / ingreso
    pmax_empresas: float = 0.25
    k_empresas: float = 4.0
    c0_empresas: float = 1.5                # carga 1 = los intereses se comen todo lo que gana
    pmax_hogares: float = 0.3
    k_hogares: float = 6.0
    c0_hogares: float = 0.6
    perdida_en_impago: float = 0.45         # el banco pierde el 45% de lo prestado (LGD de Basilea)

    # --- Reparto inicial de clientes entre bancos ---
    reparto_hogares: str = "azar"           # cada hogar a un banco al azar
    reparto_empresas: str = "azar"          # cada empresa a un banco al azar
    reparto_gobierno: str = "igual"         # la deuda pública a partes iguales entre bancos

    # --- Lista de morosos ---
    memoria_morosos: int = 10               # años que un hogar sigue en la lista tras un impago
    memoria_morosos_empresas: int = 2       # años que una empresa sigue en la lista (se reestructura antes)
    vetar_morosos: bool = True              # a un moroso no se le presta
    lista_compartida: bool = False          # E7: una lista para todos los bancos (tipo ASNEF)

    # --- Tipos de interés (relativos al tipo base del banco central) ---
    margen_min: float = 0.01                # préstamo como mínimo a base + 1%
    margen_max: float = 0.08                # préstamo como máximo a base + 8%
    umbral_max: float = 4.0                 # el umbral (deuda/ingreso) va de 0 a 4 años de ingreso
    prima_max: float = 0.05                 # extra por riesgo: de 0 a +5% por cada "vez" más endeudado que el típico
    tipo_max_sobre_base: float = 0.15       # techo: nadie paga más de base + 15% (como un límite de usura)

    # --- Lo que ve cada banco ---
    ver_rivales: bool = True                # E6: ve o no los tipos de los demás bancos

    # --- Reparto de la demanda de crédito entre bancos ---
    temperatura_softmax: float = 0.01       # más baja = el más barato se lo lleva casi todo (a ojo, por ajustar)

    # --- Recompensa ---
    lam_quiebra: float = 2.0                # penalización por quebrar, en unidades de capital inicial (E4)
    recompensa: str = "capital"             # "capital": cambio del capital / capital inicial (con signo)
                                            # "sigmoide": la del original, en (0, 1)

    # --- Entrenamiento ---
    episode_length: int = 30               # años; la economía del original se hunde hacia el año 70
    n_epochs: int = 2000
    seed: int = 1

    @classmethod
    def pruebas(cls, **cambios):
        """Economía pequeña (100 hogares, 5 empresas) para programar y probar rápido."""
        return replace(cls(n_hogares=100, n_empresas=5), **cambios)


# Valores que se barren en los experimentos del plan
SEEDS = [1, 2, 3, 4, 5]
N_BANCOS_E2 = [2, 3]                        # E2: IPPO frente a MAPPO
N_BANCOS_E6 = [1, 2, 3]                     # E6: monopolio frente a competencia
LAMBDAS_E4 = [0.0, 0.5, 2.0, 10.0]          # E4: castigo por quiebra
N_HOGARES_E8 = [100, 1000]                  # E8: generalización
