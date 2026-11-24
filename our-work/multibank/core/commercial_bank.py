"""Banco comercial: la parte que más vamos a cambiar.

Original: EconGym-original/entities/bank.py (class Bank)
- get_action()       -> la acción es (tipo de préstamo, tipo de depósito);
                        con banco central se recortan a base+1..3% y base-1..0% (bank.py:41-42).
- step()             -> beneficio = tipo_préstamo*(K empresa + deuda pública) - tipo_depósito*ahorro hogares.
- compute_next_kt()  -> presta como mucho depósitos*(1 - reservas); primero al gobierno, luego a la empresa.
- get_reward()       -> sigmoide del log del beneficio, en (0, 1).
- is_terminal()      -> siempre False: el banco nunca quiebra.

Lo nuestro:
- BalanceBancario: préstamos, depósitos y capital con doble entrada. Ya hecho y con tests.
- RegistroMorosos: la lista de morosos de cada banco. Cada impago queda apuntado; mientras
  un cliente es moroso (impago en los últimos `memoria` pasos) el banco no le presta.
- CommercialBank: hereda del original y lleva su BalanceBancario. Fase 2: usar el balance en
  step(), requisito de capital en vez de reservas, quiebra en is_terminal(), tipos hasta base+8%.
"""

from entities.bank import Bank


class RegistroMorosos:
    """La lista de morosos de un banco.

    Un cliente es moroso si ha dejado de pagar en los últimos `memoria` pasos.
    Pasado ese tiempo sin nuevos impagos, sale de la lista (como en un fichero real),
    pero su historial completo se guarda.
    """

    def __init__(self, memoria=10):
        self.memoria = memoria
        self.historial = {}                 # {cliente: [(paso, importe), ...]}

    def apuntar(self, cliente, importe, paso):
        self.historial.setdefault(cliente, []).append((paso, importe))

    def es_moroso(self, cliente, paso):
        return any(paso - p < self.memoria for p, _ in self.historial.get(cliente, []))

    def morosos(self, paso):
        """Clientes que ahora mismo están en la lista."""
        return [c for c in self.historial if self.es_moroso(c, paso)]

    def impagado_reciente(self, paso):
        """Cuánto le han dejado de pagar en los últimos `memoria` pasos."""
        return sum(x for impagos in self.historial.values() for p, x in impagos if paso - p < self.memoria)

    def veces(self, cliente):
        return len(self.historial.get(cliente, []))

class BalanceBancario:
    """El libro de cuentas de un banco: contabilidad de doble entrada (opción B del plan).

    Balance del banco:
        ACTIVO  = reservas + préstamos
        PASIVO  = depósitos
        CAPITAL = activo - pasivo   (lo que es "del banco")

    Cada operación mueve siempre DOS partidas, así el balance siempre cuadra.
    """

    def __init__(self, capital_inicial, capital_minimo=0.08, memoria_morosos=10, vetar_morosos=True):
        self.capital = capital_inicial      # dinero propio del banco
        self.prestamos = {}                 # activo: {cliente: lo que le debe}
        self.depositos = {}                 # pasivo: {cliente: lo que tiene depositado}
        self.capital_minimo = capital_minimo
        self.reservas = capital_inicial     # activo: el dinero propio guardado
        self.morosos = RegistroMorosos(memoria_morosos)
        self.vetar_morosos = vetar_morosos  # True: a un moroso no se le presta
        self.paso = 0

    def nuevo_paso(self):
        self.paso += 1

    def total_prestamos(self):
        return sum(self.prestamos.values())

    def total_depositos(self):
        return sum(self.depositos.values())

    def puede_prestar(self, x):
        # Requisito de capital: capital >= 8% de todo lo prestado (contando el nuevo préstamo)
        return self.capital >= self.capital_minimo * (self.total_prestamos() + x)

    def prestar(self, cliente, x):
        """Crea dinero: préstamo +x y depósito +x. Devuelve False si no puede."""
        if x <= 0 or self.quebrado() or not self.puede_prestar(x):
            return False
        if self.vetar_morosos and self.morosos.es_moroso(cliente, self.paso):
            return False
        self.prestamos[cliente] = self.prestamos.get(cliente, 0.0) + x
        self.depositos[cliente] = self.depositos.get(cliente, 0.0) + x
        return True

    def cobrar_intereses(self, cliente, tipo):
        """El cliente paga intereses desde su depósito: depósito −i, capital +i.

        Si no tiene suficiente en el depósito, paga lo que puede.
        Devuelve lo que se ha cobrado.
        """
        intereses = self.prestamos.get(cliente, 0.0) * tipo
        pagado = min(intereses, self.depositos.get(cliente, 0.0))
        self.depositos[cliente] = self.depositos.get(cliente, 0.0) - pagado
        self.capital += pagado
        return pagado

    def devolver(self, cliente, x):
        """Destruye dinero: préstamo −x y depósito −x."""
        deuda = self.prestamos.get(cliente, 0.0)
        saldo = self.depositos.get(cliente, 0.0)
        if x > deuda or x > saldo:
            raise ValueError(f"{cliente} no puede devolver {x} (debe {deuda}, tiene {saldo})")
        self.prestamos[cliente] = deuda - x
        self.depositos[cliente] = saldo - x

    def impago(self, cliente, x):
        """El cliente no paga x: préstamo −x y capital −x (lo pierde el banco). Entra en la lista de morosos."""
        deuda = self.prestamos.get(cliente, 0.0)
        x = min(x, deuda)
        if x <= 0:
            return
        self.prestamos[cliente] = deuda - x
        self.capital -= x
        self.morosos.apuntar(cliente, x, self.paso)

    def quebrado(self):
        return self.capital < 0

    def balance_cuadra(self, tol=1e-9):
        """Test de conservación: activo = pasivo + capital."""
        activo = self.reservas + self.total_prestamos()
        return abs(activo - (self.total_depositos() + self.capital)) < tol


class CommercialBank(Bank):
    """El Bank del original, usado como "banco agregado" de la economía.

    Los N bancos de verdad viven en multibank/banking.py (SistemaBancario). Antes de cada paso,
    el entorno le pasa aquí los tipos medios de esos bancos (fijar_tipos), y la economía del
    original los usa en lugar de la acción de su propio agente de banco.
    """

    def __init__(self, entity_args, capital_inicial=100.0, capital_minimo=0.08):
        super().__init__(entity_args)
        self.balance = BalanceBancario(capital_inicial, capital_minimo)
        self.tipos_fijados = None

    def reset(self, **custom_cfg):
        # Igual que en el banco central: el original no restaura estos dos valores entre episodios.
        if not hasattr(self, "inicial"):
            self.inicial = (self.base_interest_rate, self.reserve_ratio)
        self.base_interest_rate, self.reserve_ratio = self.inicial
        self.tipos_fijados = None
        super().reset(**custom_cfg)

    def fijar_tipos(self, prestamo, deposito):
        self.tipos_fijados = (float(prestamo), float(deposito))

    def get_action(self, actions, central_bank_exist=False):
        if self.tipos_fijados is None:
            return super().get_action(actions, central_bank_exist)
        self.lending_rate, self.deposit_rate = self.tipos_fijados
