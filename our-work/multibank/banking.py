"""El sistema bancario: N bancos que prestan a empresas y hogares, encima de la economía del original.

La economía de EconGym (producción, salarios, ahorro de los hogares, gobierno, banco central)
sigue funcionando como siempre. Este módulo añade los bancos de verdad:

- Clientes: F empresas prestatarias + H hogares. Las empresas son trozos de la empresa agregada
  del original, cada una con su tamaño y su propia suerte (shock) cada año.
- Cada paso: cobrar intereses -> impagos -> pagar depósitos -> amortizar y refinanciar los
  préstamos (reparto softmax por tipos, lista de morosos, umbral, requisito de capital)
  -> mover depósitos -> quiebras.

Balance de cada banco (siempre cuadra, se comprueba en cada paso):
    ACTIVO  = préstamos + bonos del gobierno + liquidez
    PASIVO  = depósitos
    CAPITAL = activo - pasivo
La liquidez es el dinero en el banco central: rinde el tipo base (y si es negativa, paga el tipo base).

Arrays: los préstamos son una matriz L[banco, cliente]; los clientes 0..F-1 son empresas y F.. hogares.
"""

from dataclasses import dataclass

import numpy as np

NUNCA = -10**9


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Tipos:
    """Lo que decide cada banco en un paso (arrays de tamaño N)."""
    deposito: np.ndarray
    empresas: np.ndarray
    hogares: np.ndarray
    prima: np.ndarray         # extra por riesgo: cuánto más cobra por cada "vez" que el cliente va más endeudado que el típico
    umbral: np.ndarray        # deuda/ingreso máxima de un hogar al que se acepta prestar
    techo: float = np.inf     # tipo máximo que se puede cobrar a nadie (base + tipo_max_sobre_base)


class SistemaBancario:

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.N = cfg.n_bancos

    # ------------------------------------------------------------------ inicio
    def iniciar(self, econ):
        cfg, rng, N = self.cfg, self.rng, self.N
        h, m = econ.households, econ.market
        self.F, self.H = cfg.n_empresas, h.households_n
        F, H = self.F, self.H
        self.C = F + H
        self.es_empresa = np.r_[np.ones(F, bool), np.zeros(H, bool)]

        # Empresas prestatarias: tamaños desiguales (como en la realidad) y suerte propia
        self.peso_empresa = rng.lognormal(0.0, 1.0, F)
        self.peso_empresa /= self.peso_empresa.sum()
        self.suerte = np.ones(F)

        # Deuda que quiere cada cliente
        K_total = float(np.sum(m.Kt))
        deuda_hog = np.ravel(h.debt_init) if cfg.deuda_real else np.zeros(H)
        self.objetivo = np.r_[cfg.apalancamiento_empresas * K_total * self.peso_empresa, deuda_hog].astype(float)
        self.amortizacion = np.where(self.es_empresa, cfg.amortizacion_empresas, cfg.amortizacion_hogares)

        # Reparto inicial: cada cliente a un banco al azar
        banco_de = rng.integers(0, N, self.C)
        self.L = np.zeros((N, self.C))
        self.L[banco_de, np.arange(self.C)] = self.objetivo

        # Depósitos: la parte de los activos de cada hogar que está en el banco (LIQ), en un banco al azar
        self.parte_en_banco = np.ravel(h.liq_share)
        at = np.maximum(np.ravel(h.at), 0.0) * self.parte_en_banco
        banco_dep = rng.integers(0, N, H)
        self.Dep = np.bincount(banco_dep, weights=at, minlength=N).astype(float)

        # Deuda pública a partes iguales
        self.G = np.full(N, max(float(econ.main_gov.Bt), 0.0) / N)

        # Capital inicial = 15% de lo que presta cada banco; la liquidez cuadra el balance
        self.K = cfg.capital_inicial * self.L.sum(axis=1)
        self.K0 = self.K.copy()
        self.Liq = self.K + self.Dep - self.L.sum(axis=1) - self.G

        self.riesgo = np.ones(self.C)                       # 1 = tan endeudado como el cliente típico
        self.ultimo_impago = np.full((N, self.C), NUNCA)
        self.quebrado = np.zeros(N, bool)
        self.t = 0
        self.tipos = None
        self.ultimo = {}
        self.comprobar_balance()

    # ------------------------------------------------------------------ utilidades
    def activo(self):
        return self.L.sum(axis=1) + self.G + self.Liq

    def comprobar_balance(self, tol=1e-9):
        """Test de conservación: activo = depósitos + capital, banco a banco."""
        resto = self.activo() - (self.Dep + self.K)
        escala = np.maximum(1.0, np.abs(self.activo()))
        if np.any(np.abs(resto) / escala > tol):
            raise AssertionError(f"El balance no cuadra: {resto}")

    def es_moroso(self):
        """(N, C): True si el cliente está en la lista de morosos de ese banco."""
        memoria = np.where(self.es_empresa, self.cfg.memoria_morosos_empresas, self.cfg.memoria_morosos)
        reciente = (self.t - self.ultimo_impago) < memoria[None, :]
        if self.cfg.lista_compartida:
            reciente = np.broadcast_to(reciente.any(axis=0), reciente.shape)
        return reciente

    def tipo_por_cliente(self, tipos):
        """(N, C): el tipo que cobra cada banco a cada cliente.

        tipo = base + margen de su clase (empresa u hogar) + prima × (riesgo - 1), sin pasar del techo.
        Un cliente típico (riesgo 1) o más seguro paga solo el margen; uno el doble de endeudado, + prima.
        """
        base = np.where(self.es_empresa[None, :], tipos.empresas[:, None], tipos.hogares[:, None])
        extra = tipos.prima[:, None] * np.maximum(self.riesgo - 1.0, 0.0)[None, :]
        return np.minimum(base + extra, tipos.techo)

    def calcular_riesgo(self, econ, excedente_emp):
        """Riesgo de cada cliente frente al típico de su clase (mediana = 1).
        Empresa: su deuda / su beneficio del año. Hogar: su deuda / su ingreso."""
        F = self.F
        x = np.r_[self.objetivo[:F] / excedente_emp, self.deuda_ingreso(econ)]
        riesgo = np.zeros(self.C)
        for clase in (slice(0, F), slice(F, self.C)):
            v = x[clase]
            positivos = v[v > 0]
            if positivos.size:
                riesgo[clase] = v / np.median(positivos)
        self.riesgo = riesgo

    def deuda_ingreso(self, econ):
        ingreso = np.maximum(np.ravel(econ.households.income), self.cfg.ingreso_minimo)
        return self.objetivo[self.F:] / ingreso

    # ------------------------------------------------------------------ un paso
    def paso(self, econ, tipos: Tipos):
        cfg, rng, N, F = self.cfg, self.rng, self.N, self.F
        m, h = econ.market, econ.households
        self.t += 1
        self.tipos = tipos
        K_antes = self.K.copy()
        base = float(econ.main_gov.base_interest_rate)
        vivo = ~self.quebrado

        # Suerte de cada empresa este año (persistente)
        self.suerte = np.exp(cfg.persistencia_suerte * np.log(self.suerte)
                             + rng.normal(0, cfg.volatilidad_suerte, F))

        excedente = float(np.sum(m.price * m.Yt_j - m.WageRate * m.firm_labor_j))
        excedente_emp = np.maximum(excedente * self.peso_empresa * self.suerte, 1e-8)
        self.calcular_riesgo(econ, excedente_emp)

        # --- 1. Intereses e impagos sobre lo prestado el paso anterior ---
        r = self.tipo_por_cliente(tipos)
        intereses = self.L * r                               # (N, C)
        int_cliente = intereses.sum(axis=0)
        deuda_cliente = self.L.sum(axis=0)
        carga_emp = int_cliente[:F] / excedente_emp
        ingreso = np.maximum(np.ravel(h.income), cfg.ingreso_minimo)
        carga_hog = (int_cliente[F:] + self.amortizacion[F:] * deuda_cliente[F:]) / ingreso

        p = np.r_[cfg.pmax_empresas * sigmoide(cfg.k_empresas * (carga_emp - cfg.c0_empresas)),
                  cfg.pmax_hogares * sigmoide(cfg.k_hogares * (carga_hog - cfg.c0_hogares))]
        impaga = (rng.random(self.C) < p) & (deuda_cliente > 0)

        cobrado = (intereses * ~impaga[None, :]).sum(axis=1)
        self.Liq += cobrado
        self.K += cobrado

        expuesto = self.L * impaga[None, :]
        perdida = cfg.perdida_en_impago * expuesto.sum(axis=1)
        recuperado = expuesto.sum(axis=1) - perdida
        self.K -= perdida
        self.Liq += recuperado
        self.ultimo_impago[expuesto > 0] = self.t
        self.L[:, impaga] = 0.0

        # --- 2. Intereses de los depósitos y rendimiento de bonos y liquidez ---
        pagado_dep = tipos.deposito * self.Dep
        self.Dep += pagado_dep
        self.K -= pagado_dep
        rinde = base * (self.Liq + self.G)
        self.Liq += rinde
        self.K += rinde

        # --- 3. Amortizar y refinanciar ---
        # Lo que quieren pedir las empresas este año: el 70% del capital que la economía quiere invertir
        K_deseado = float(np.sum(m.Kt_next))
        self.objetivo[:F] = cfg.apalancamiento_empresas * K_deseado * self.peso_empresa
        L_am = self.L * (1 - self.amortizacion)[None, :]
        self.Liq += (self.L - L_am).sum(axis=1)            # lo devuelto vuelve al banco
        acepta = vivo[:, None] & np.ones((N, self.C), bool)
        if cfg.vetar_morosos:
            acepta &= ~self.es_moroso()
        dti = self.deuda_ingreso(econ)
        acepta[:, F:] &= dti[None, :] <= tipos.umbral[:, None]

        congelado = L_am * ~acepta                          # los bancos que rechazan no renuevan
        abierto = L_am * acepta
        necesita = np.maximum(self.objetivo - congelado.sum(axis=0), 0.0)
        # Reparto softmax: el más barato se lleva más, pero no todo
        puntos = np.where(acepta, -r / cfg.temperatura_softmax, -np.inf)
        puntos -= np.where(np.isfinite(puntos.max(axis=0)), puntos.max(axis=0), 0.0)[None, :]
        pesos = np.where(acepta, np.exp(puntos), 0.0)
        suma = pesos.sum(axis=0)
        cuota = np.divide(pesos, suma[None, :], out=np.zeros_like(pesos), where=suma[None, :] > 0)
        deseado = cuota * necesita[None, :]

        # Requisito de capital: préstamos totales <= capital / capital_minimo
        mantiene = np.minimum(deseado, abierto)
        nuevo = deseado - mantiene
        capacidad = np.where(self.K > 0, self.K / cfg.capital_minimo, 0.0)
        margen = np.maximum(capacidad - congelado.sum(axis=1) - mantiene.sum(axis=1), 0.0)
        pedido = nuevo.sum(axis=1)
        factor = np.divide(margen, pedido, out=np.ones(N), where=pedido > 0).clip(0, 1)
        L_nuevo = congelado + mantiene + nuevo * factor[:, None]

        self.Liq -= (L_nuevo - L_am).sum(axis=1)            # prestar saca liquidez, cobrar la mete
        self.L = L_nuevo
        concedido = self.L.sum(axis=0)
        racionado = np.maximum(self.objetivo - concedido, 0.0)

        # --- 4. Depósitos: los hogares los mueven según el tipo que paga cada banco ---
        total_dep = float(np.sum(np.maximum(np.ravel(h.at_next), 0.0) * self.parte_en_banco))
        puntos_d = np.where(vivo, tipos.deposito / cfg.temperatura_softmax, -np.inf)
        if vivo.any():
            puntos_d = puntos_d - puntos_d[vivo].max()
        pesos_d = np.where(vivo, np.exp(puntos_d), 0.0)
        Dep_obj = pesos_d / max(pesos_d.sum(), 1e-12) * total_dep
        self.Liq += Dep_obj - self.Dep
        self.Dep = Dep_obj

        # Deuda pública a partes iguales entre los bancos vivos
        n_vivos = max(int(vivo.sum()), 1)
        G_obj = np.where(vivo, max(float(econ.main_gov.Bt_next), 0.0) / n_vivos, 0.0)
        self.Liq -= G_obj - self.G
        self.G = G_obj

        # --- 5. Quiebras ---
        nuevas_quiebras = vivo & (self.K < 0)
        self.quebrado |= nuevas_quiebras

        self.comprobar_balance()

        # Lo que se lleva cada empresa: si le falta crédito, invierte menos
        credito_empresas = concedido[:F].sum()
        self.ultimo = dict(
            beneficio=self.K - K_antes, perdida=perdida, cobrado=cobrado, pagado_dep=pagado_dep,
            impaga=impaga, p_impago=p, racionado=racionado, nuevas_quiebras=nuevas_quiebras,
            credito_empresas=credito_empresas, base=base, excedente_emp=excedente_emp,
            riesgo=self.riesgo.copy(), tipo_cliente=r,
        )
        return self.ultimo

    # ------------------------------------------------------------------ resúmenes
    def cuota_prestamos(self):
        tot_e, tot_h = self.L[:, :self.F].sum(), self.L[:, self.F:].sum()
        return (self.L[:, :self.F].sum(axis=1) / max(tot_e, 1e-12),
                self.L[:, self.F:].sum(axis=1) / max(tot_h, 1e-12))

    def ratio_capital(self):
        return self.K / np.maximum(self.L.sum(axis=1), 1e-12)

    def tipo_medio(self):
        """Tipos medios ponderados (para que la economía del original los use)."""
        t = self.tipos
        L_emp = self.L[:, :self.F]
        prest = float(np.sum(self.tipo_por_cliente(t)[:, :self.F] * L_emp) / max(L_emp.sum(), 1e-12))
        dep = float(np.sum(t.deposito * self.Dep) / max(self.Dep.sum(), 1e-12))
        return prest, dep
