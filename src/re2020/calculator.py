"""Évaluation arithmétique sûre : analyse de l'arbre syntaxique, aucun appel à eval."""

from __future__ import annotations

import ast
import math
import operator

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"min": min, "max": max, "abs": abs, "round": round, "sqrt": math.sqrt}
MAX_EXPR_LEN = 300
MAX_EXPONENT = 100


class CalculError(ValueError):
    """Expression refusée ou invalide."""


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculError("exposant trop grand")
        try:
            return _BINOPS[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalculError("division par zéro") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise CalculError(f"élément non autorisé : {type(node).__name__}")


def calculate(expression: str) -> float:
    """Évalue une expression arithmétique. Accepte la virgule décimale française (« 0,15 »)."""
    expr = expression.strip()
    if not expr or len(expr) > MAX_EXPR_LEN:
        raise CalculError("expression vide ou trop longue")
    expr = _french_decimals(expr).replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculError(f"syntaxe invalide : {exc.msg}") from exc
    result = _eval(tree)
    if isinstance(result, float) and not math.isfinite(result):
        raise CalculError("résultat non fini")
    return result


def _french_decimals(expr: str) -> str:
    """Remplace « 12,5 » par « 12.5 » sans toucher aux virgules séparant des arguments (« min(1, 2) »)."""
    out = []
    for i, ch in enumerate(expr):
        if ch == "," and 0 < i < len(expr) - 1 and expr[i - 1].isdigit() and expr[i + 1].isdigit():
            out.append(".")
        else:
            out.append(ch)
    return "".join(out)
