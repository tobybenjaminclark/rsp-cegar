from __future__ import annotations

from functools import reduce
import re

from .ast import And
from .ast import BooleanExpr
from .ast import Cmp
from .ast import CmpOp
from .ast import Not
from .ast import Or
from .ast import Symbol
from .verifier import CompleteOrderVerifier
from synthesis.synth import SynthesisResult


def and_all(terms: list[BooleanExpr]) -> BooleanExpr:
    if not terms:
        raise ValueError("and_all requires at least one term.")
    deduplicated = []
    seen = set()
    for term in _flatten_and_terms(terms):
        key = str(term)
        if key not in seen:
            deduplicated.append(term)
            seen.add(key)
    return reduce(lambda left, right: And(left, right), deduplicated)


def _flatten_and_terms(terms: list[BooleanExpr]) -> list[BooleanExpr]:
    flattened = []
    for term in terms:
        if isinstance(term, And):
            flattened.extend(_flatten_and_terms([term.left, term.right]))
        else:
            flattened.append(term)
    return flattened


def seed_from_synthesis_result(
    verifier: CompleteOrderVerifier,
    result: SynthesisResult,
) -> BooleanExpr:
    if result.rule_solution is None:
        raise RuntimeError(f"SyGuS did not find a seed rule: {result.check}")

    schema = _extract_define_fun_body(result.rule_solution)
    return and_all([
        _sexpr_to_cegis_ast(_expand_lets(schema), aircraft=aircraft)
        for aircraft in verifier.aircraft
    ])


def _extract_define_fun_body(solution: str):
    define_fun = _parse_sexpr(solution)
    while isinstance(define_fun, list) and len(define_fun) == 1:
        define_fun = define_fun[0]
    if len(define_fun) < 5 or define_fun[0] != "define-fun":
        raise ValueError(f"Unexpected SyGuS solution shape: {solution}")
    return define_fun[4]


def _parse_sexpr(text: str):
    tokens = re.findall(r"\(|\)|\|[^|]*\||[^\s()]+", text)
    stack: list[list] = [[]]
    for token in tokens:
        if token == "(":
            child: list = []
            stack[-1].append(child)
            stack.append(child)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError(f"Unbalanced SyGuS solution: {text}")
            stack.pop()
        else:
            stack[-1].append(token[1:-1] if token.startswith("|") and token.endswith("|") else token)
    if len(stack) != 1:
        raise ValueError(f"Unbalanced SyGuS solution: {text}")
    return stack[0]


def _expand_lets(expr, env=None):
    env = dict(env or {})
    if isinstance(expr, str):
        return env.get(expr, expr)
    if not expr:
        return expr
    if expr[0] != "let":
        return [_expand_lets(part, env) for part in expr]

    local_env = dict(env)
    for name, value in expr[1]:
        local_env[name] = _expand_lets(value, local_env)
    return _expand_lets(expr[2], local_env)


def _sexpr_to_cegis_ast(expr, *, aircraft: str) -> BooleanExpr:
    if not isinstance(expr, list):
        raise ValueError(f"Expected Boolean expression, got: {expr}")
    head, *args = expr

    if head == "and":
        return and_all([_sexpr_to_cegis_ast(arg, aircraft=aircraft) for arg in args])
    if head == "or":
        return reduce(lambda left, right: Or(left, right), [_sexpr_to_cegis_ast(arg, aircraft=aircraft) for arg in args])
    if head == "not":
        return Not(_sexpr_to_cegis_ast(args[0], aircraft=aircraft))
    if head in ("<=", ">=", "<", ">", "="):
        return Cmp(_sexpr_to_symbol(args[0], aircraft), _cmp_op(head), _sexpr_to_symbol(args[1], aircraft))
    raise ValueError(f"Unsupported SyGuS expression: {expr}")


def _cmp_op(op: str) -> CmpOp:
    return {
        "<=": CmpOp.LE,
        ">=": CmpOp.GE,
        "<": CmpOp.LT,
        ">": CmpOp.GT,
        "=": CmpOp.EQ,
    }[op]


def _sexpr_to_symbol(expr, aircraft: str) -> Symbol:
    if not isinstance(expr, str):
        raise ValueError(f"Expected symbol, got: {expr}")
    return Symbol(_instantiate_schema_symbol(expr, aircraft))


def _instantiate_schema_symbol(name: str, aircraft: str) -> str:
    if name == "D_i_x":
        return f"D_i_{aircraft}"
    if name == "D_j_x":
        return f"D_j_{aircraft}"
    if name == "D_x_i":
        return f"D_{aircraft}_i"
    if name == "D_x_j":
        return f"D_{aircraft}_j"
    return name
