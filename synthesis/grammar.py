from __future__ import annotations

from dataclasses import dataclass

from cvc5 import Kind

from core.context import RSPContext


@dataclass(frozen=True)
class SygusSymbol:
    name: str
    formal: object
    actual: object | None


def make_allowed_symbols(ctx: RSPContext) -> tuple[SygusSymbol, ...]:
    solver = ctx.solver
    symbols: list[SygusSymbol] = []
    seen: set[str] = set()

    def add_symbol(name: str, actual) -> None:
        if name in seen:
            return
        seen.add(name)
        symbols.append(SygusSymbol(name, solver.mkVar(ctx.real_sort, name), actual))

    add_symbol("R_i", ctx.r["i"])
    add_symbol("R_j", ctx.r["j"])
    add_symbol("LT_i", ctx.lt["i"])
    add_symbol("LT_j", ctx.lt["j"])
    add_symbol("LC_i", ctx.lc["i"])
    add_symbol("LC_j", ctx.lc["j"])
    add_symbol("ET_i", ctx.et["i"])
    add_symbol("ET_j", ctx.et["j"])
    add_symbol("EC_i", ctx.ec["i"])
    add_symbol("EC_j", ctx.ec["j"])
    add_symbol("B_i", ctx.b["i"])
    add_symbol("B_j", ctx.b["j"])
    add_symbol("C_i", ctx.c["i"])
    add_symbol("C_j", ctx.c["j"])
    add_symbol("D_i_x", None)
    add_symbol("D_j_x", None)
    add_symbol("D_x_i", None)
    add_symbol("D_x_j", None)

    return tuple(symbols)


def symbol_map(symbols: tuple[SygusSymbol, ...]) -> dict[str, object]:
    return {symbol.name: symbol.formal for symbol in symbols}


def allowed_predicates(env, symbols: tuple[SygusSymbol, ...]) -> list[object]:
    solver = env.solver
    formals = symbol_map(symbols)
    predicates = []

    for prefix in ("R", "LT", "LC", "ET", "EC", "B", "C"):
        left = formals.get(f"{prefix}_i")
        right = formals.get(f"{prefix}_j")
        if left is None or right is None:
            continue
        predicates.append(solver.mkTerm(Kind.LEQ, left, right))
        predicates.append(solver.mkTerm(Kind.LEQ, right, left))
        predicates.append(solver.mkTerm(Kind.EQUAL, left, right))

    for plane in sorted({name.split("_", 2)[2] for name in formals if name.startswith("D_i_")}):
        outgoing_i = formals.get(f"D_i_{plane}")
        outgoing_j = formals.get(f"D_j_{plane}")
        incoming_i = formals.get(f"D_{plane}_i")
        incoming_j = formals.get(f"D_{plane}_j")

        if outgoing_i is not None and outgoing_j is not None:
            predicates.append(solver.mkTerm(Kind.LEQ, outgoing_i, outgoing_j))
            predicates.append(solver.mkTerm(Kind.LEQ, outgoing_j, outgoing_i))
            predicates.append(solver.mkTerm(Kind.EQUAL, outgoing_i, outgoing_j))

        if incoming_i is not None and incoming_j is not None:
            predicates.append(solver.mkTerm(Kind.LEQ, incoming_i, incoming_j))
            predicates.append(solver.mkTerm(Kind.LEQ, incoming_j, incoming_i))
            predicates.append(solver.mkTerm(Kind.EQUAL, incoming_i, incoming_j))

    return predicates


def make_pruning_rule_grammar(env, symbols: tuple[SygusSymbol, ...], max_conjuncts: int):
    if max_conjuncts < 1:
        raise ValueError("max_conjuncts must be at least 1")

    solver = env.solver
    start = solver.mkVar(env.bool_sort, "Rule")
    atom = solver.mkVar(env.bool_sort, "Atom")
    predicates = allowed_predicates(env, symbols)

    rule_shapes = [atom]
    for width in range(2, max_conjuncts + 1):
        rule_shapes.append(solver.mkTerm(Kind.AND, *([atom] * width)))

    grammar = solver.mkGrammar([symbol.formal for symbol in symbols], [start, atom])
    grammar.addRules(start, rule_shapes)
    grammar.addRules(atom, predicates)
    return grammar
