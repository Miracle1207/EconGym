import pytest

from multibank.core.commercial_bank import BalanceBancario


def test_prestar_crea_dinero():
    b = BalanceBancario(capital_inicial=100)
    assert b.prestar("empresa", 500)
    assert b.total_prestamos() == 500
    assert b.total_depositos() == 500
    assert b.balance_cuadra()


def test_requisito_de_capital():
    b = BalanceBancario(capital_inicial=100)       # 8% -> puede prestar hasta 1250
    assert b.puede_prestar(1250)
    assert not b.puede_prestar(1251)
    assert not b.prestar("empresa", 2000)
    assert b.total_prestamos() == 0


def test_devolver_destruye_dinero():
    b = BalanceBancario(capital_inicial=100)
    b.prestar("empresa", 500)
    b.devolver("empresa", 200)
    assert b.total_prestamos() == 300
    assert b.total_depositos() == 300
    assert b.balance_cuadra()


def test_no_se_devuelve_mas_de_lo_que_se_debe():
    b = BalanceBancario(capital_inicial=100)
    b.prestar("empresa", 500)
    with pytest.raises(ValueError):
        b.devolver("empresa", 600)


def test_intereses_suben_el_capital():
    b = BalanceBancario(capital_inicial=100)
    b.prestar("empresa", 500)
    cobrado = b.cobrar_intereses("empresa", 0.05)
    assert cobrado == pytest.approx(25)
    assert b.capital == pytest.approx(125)
    assert b.balance_cuadra()


def test_impago_puede_quebrar():
    b = BalanceBancario(capital_inicial=100)
    b.prestar("empresa", 500)
    b.impago("empresa", 150)
    assert b.capital == -50
    assert b.quebrado()
    assert b.balance_cuadra()
    assert not b.prestar("empresa", 1)            # quebrado: ya no presta


def test_impago_apunta_en_la_lista_de_morosos():
    b = BalanceBancario(capital_inicial=100)
    b.prestar("empresa", 500)
    b.impago("empresa", 50)
    assert b.morosos.es_moroso("empresa", b.paso)
    assert b.morosos.veces("empresa") == 1
    assert b.morosos.impagado_reciente(b.paso) == 50
    assert b.balance_cuadra()


def test_a_un_moroso_no_se_le_presta():
    b = BalanceBancario(capital_inicial=100)
    b.prestar("empresa", 500)
    b.impago("empresa", 50)
    assert not b.prestar("empresa", 10)
    assert b.prestar("hogar_3", 10)               # a los demás sí


def test_se_sale_de_la_lista_pasado_un_tiempo():
    b = BalanceBancario(capital_inicial=100, memoria_morosos=3)
    b.prestar("empresa", 500)
    b.impago("empresa", 50)
    for _ in range(3):
        b.nuevo_paso()
    assert not b.morosos.es_moroso("empresa", b.paso)
    assert b.morosos.veces("empresa") == 1        # el historial se queda
    assert b.prestar("empresa", 10)
