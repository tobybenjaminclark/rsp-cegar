from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, reduce
from itertools import product
from typing import Iterable

import cvc5
from cvc5 import Kind


PHI_SIZE = 2


def real_val(solver: cvc5.Solver, value: int | str):
    return solver.mkReal(str(value))


def num_val(solver: cvc5.Solver, sort, value: int | str):
    if sort.isInteger():
        return solver.mkInteger(int(value))
    return solver.mkReal(str(value))


def add_terms(solver: cvc5.Solver, *terms):
    if not terms:
        return real_val(solver, 0)
    if len(terms) == 1:
        return terms[0]
    return solver.mkTerm(Kind.ADD, *terms)


def mul_terms(solver: cvc5.Solver, *terms):
    if not terms:
        return real_val(solver, 1)
    if len(terms) == 1:
        return terms[0]
    return solver.mkTerm(Kind.MULT, *terms)


def zmax(solver: cvc5.Solver, left, right):
    return solver.mkTerm(
        Kind.ITE,
        solver.mkTerm(Kind.GEQ, left, right),
        left,
        right,
    )


def zmax_list(solver: cvc5.Solver, terms: Iterable):
    terms = list(terms)
    if not terms:
        raise ValueError("zmax_list requires at least one term.")
    return reduce(lambda left, right: zmax(solver, left, right), terms)


@dataclass(frozen=True)
class RSPContext:
    solver: cvc5.Solver
    real_sort: object
    bool_sort: object
    integer_arithmetic: bool
    aircraft: tuple[str, ...]
    r: dict[str, object]
    b: dict[str, object]
    c: dict[str, object]
    ec: dict[str, object]
    lc: dict[str, object]
    et: dict[str, object]
    lt: dict[str, object]
    delta: dict[tuple[str, str], object]

    @property
    def variable_constraints(self) -> list[object]:
        zero = num_val(self.solver, self.real_sort, 0)
        return [
            *[
                self.solver.mkTerm(Kind.GEQ, param[ac], zero)
                for ac in self.aircraft
                for param in (self.r, self.b, self.c, self.ec, self.lc, self.et, self.lt)
            ],
            *[
                self.solver.mkTerm(Kind.GEQ, self.delta[(x, y)], zero)
                for x, y in product(self.aircraft, self.aircraft)
            ],
        ]

    @property
    def ordered_window_constraints(self) -> list[object]:
        return [
            self.solver.mkTerm(
                Kind.AND,
                self.solver.mkTerm(Kind.LT, self.et[ac], self.lt[ac]),
                self.solver.mkTerm(Kind.LT, self.ec[ac], self.lc[ac]),
            )
            for ac in self.aircraft
        ]

    @property
    def release_time_constraints(self) -> list[object]:
        return [
            self.solver.mkTerm(
                Kind.EQUAL,
                self.r[ac],
                zmax(
                    self.solver,
                    add_terms(self.solver, self.b[ac], self.c[ac]),
                    zmax(self.solver, self.et[ac], self.ec[ac]),
                ),
            )
            for ac in self.aircraft
        ]

    @cached_property
    def foundational_constraints(self) -> list[object]:
        return (
            self.variable_constraints
            + self.ordered_window_constraints
            + self.release_time_constraints
        )

    def separation_equivalence(self, i: str, j: str) -> list[object]:
        return [
            self.solver.mkTerm(
                Kind.AND,
                self.solver.mkTerm(Kind.EQUAL, self.delta[(i, x)], self.delta[(j, x)]),
                self.solver.mkTerm(Kind.EQUAL, self.delta[(x, i)], self.delta[(x, j)]),
            )
            for x in self.aircraft
        ]

    def with_sequence(self, seq) -> "RSPSequenceContext":
        return RSPSequenceContext(self, tuple(seq))


def make_context(
    aircraft: list[str] | tuple[str, ...],
    solver: cvc5.Solver | None = None,
    configure_solver: bool = True,
    integer_arithmetic: bool = False,
    use_sygus_vars: bool = False,
) -> RSPContext:
    solver = solver or cvc5.Solver()
    if configure_solver:
        solver.setLogic("QF_LIA" if integer_arithmetic else "QF_LRA")
        solver.setOption("produce-models", "true")
    real_sort = solver.getIntegerSort() if integer_arithmetic else solver.getRealSort()
    bool_sort = solver.getBooleanSort()
    aircraft = tuple(aircraft)
    if use_sygus_vars:
        def mk_symbol(name: str):
            return solver.declareSygusVar(name, real_sort)
    else:
        def mk_symbol(name: str):
            return solver.mkConst(real_sort, name)
    return RSPContext(
        solver=solver,
        real_sort=real_sort,
        bool_sort=bool_sort,
        integer_arithmetic=integer_arithmetic,
        aircraft=aircraft,
        r={ac: mk_symbol(f"r_{ac}") for ac in aircraft},
        b={ac: mk_symbol(f"b_{ac}") for ac in aircraft},
        c={ac: mk_symbol(f"c_{ac}") for ac in aircraft},
        ec={ac: mk_symbol(f"ec_{ac}") for ac in aircraft},
        lc={ac: mk_symbol(f"lc_{ac}") for ac in aircraft},
        et={ac: mk_symbol(f"et_{ac}") for ac in aircraft},
        lt={ac: mk_symbol(f"lt_{ac}") for ac in aircraft},
        delta={(x, y): mk_symbol(f"d_{x}_{y}") for x, y in product(aircraft, aircraft)},
    )


@dataclass(frozen=True)
class RSPSequenceContext:
    ctx: RSPContext
    seq: tuple[str, ...]

    def __post_init__(self) -> None:
        if set(self.seq) != set(self.ctx.aircraft):
            raise ValueError("Sequence must be a permutation of base aircraft.")

    @cached_property
    def takeoff(self) -> dict[str, object]:
        solver = self.ctx.solver
        seq = self.seq
        takeoff = {seq[0]: self.ctx.r[seq[0]]}

        for index in range(1, len(seq)):
            plane = seq[index]
            predecessor_terms = [
                add_terms(solver, takeoff[x], self.ctx.delta[(x, plane)])
                for x in seq[:index]
            ]
            takeoff[plane] = zmax(
                solver,
                self.ctx.r[plane],
                zmax_list(solver, predecessor_terms),
            )

        return takeoff

    @cached_property
    def delay(self) -> dict[str, object]:
        solver = self.ctx.solver
        return {
            ac: solver.mkTerm(Kind.SUB, self.takeoff[ac], self.ctx.b[ac])
            for ac in self.seq
        }

    @cached_property
    def ctot(self) -> dict[str, object]:
        return {
            ac: ctot_cost(self.ctx, self.takeoff[ac], ac)
            for ac in self.seq
        }

    @cached_property
    def makespan(self):
        return zmax_list(self.ctx.solver, [self.takeoff[x] for x in self.ctx.aircraft])

    @cached_property
    def window_violation(self) -> dict[str, object]:
        solver = self.ctx.solver
        return {
            ac: solver.mkTerm(Kind.GT, self.takeoff[ac], self.ctx.lt[ac])
            for ac in self.seq
        }

    @cached_property
    def time_window_feasible(self) -> list[object]:
        solver = self.ctx.solver
        return [
            solver.mkTerm(Kind.LEQ, self.takeoff[ac], self.ctx.lt[ac])
            for ac in self.seq
        ]


def ctot_cost(ctx: RSPContext, takeoff, aircraft: str):
    solver = ctx.solver
    lc = ctx.lc[aircraft]
    late = solver.mkTerm(Kind.SUB, takeoff, lc)
    zero = num_val(solver, ctx.real_sort, 0)
    two = num_val(solver, ctx.real_sort, 2)
    three = num_val(solver, ctx.real_sort, 3)
    four = num_val(solver, ctx.real_sort, 4)
    three_hundred = num_val(solver, ctx.real_sort, 300)
    return solver.mkTerm(
        Kind.ITE,
        solver.mkTerm(Kind.LEQ, takeoff, lc),
        zero,
        solver.mkTerm(
            Kind.ITE,
            solver.mkTerm(Kind.LEQ, takeoff, add_terms(solver, lc, three_hundred)),
            add_terms(solver, late, two),
            add_terms(solver, mul_terms(solver, three, late), four),
        ),
    )


SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sub(n: int) -> str:
    return str(n).translate(SUBSCRIPTS)


def make_phi_block(k: int) -> list[str]:
    return [f"psi{sub(k)}{sub(t)}" for t in range(1, PHI_SIZE + 1)]


def get_sequences(integer_arithmetic: bool = False) -> tuple[RSPSequenceContext, RSPSequenceContext, RSPContext]:
    psi1, psi2, psi3 = make_phi_block(1), make_phi_block(2), make_phi_block(3)
    ctx = make_context(psi1 + ["i"] + psi2 + ["j"] + psi3, integer_arithmetic=integer_arithmetic)
    s1 = ctx.with_sequence(psi1 + ["i"] + psi2 + ["j"] + psi3)
    s2 = ctx.with_sequence(psi1 + ["j"] + psi2 + ["i"] + psi3)
    return s1, s2, ctx
