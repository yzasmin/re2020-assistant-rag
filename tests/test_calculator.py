import math

import pytest

from re2020.calculator import CalculError, calculate


def test_operations_de_base():
    assert calculate("63 * 1,15 - 3,6") == pytest.approx(68.85)
    assert calculate("2500 * 2,3") == 5750
    assert calculate("250 / 740") == pytest.approx(0.337837, abs=1e-6)


def test_fonctions_autorisees():
    assert calculate("min(2,9; 3)".replace(";", ",")) == 2.9
    assert calculate("round(0,3378 * 100)") == 34
    assert calculate("sqrt(16)") == 4


def test_virgule_francaise_et_separateur_d_arguments():
    assert calculate("max(1,5, 2)") == 2
    assert calculate("0,6 * 1,2") == pytest.approx(0.72)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('dir')",
    "open('secret.txt').read()",
    "[x for x in range(10)]",
    "lambda: 1",
    "1 if True else 2",
    "variable + 1",
    "2 ** 100000",
    "1/0",
    "",
    "3 +",
])
def test_expressions_refusees(expression):
    with pytest.raises(CalculError):
        calculate(expression)


def test_pas_de_resultat_non_fini():
    with pytest.raises(CalculError):
        calculate("1e308 * 10")


def test_longueur_maximale():
    with pytest.raises(CalculError):
        calculate("1+" * 200 + "1")


def test_resultat_est_un_nombre():
    valeur = calculate("(740 - 490) / 740")
    assert isinstance(valeur, float) and math.isfinite(valeur)
