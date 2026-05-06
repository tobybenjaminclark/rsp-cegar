from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

import cvc5
from cvc5 import Kind

from core.checks import Counterexample
from core.checks import Unverified
from core.checks import Verified
from core.checks import verify_pruning_rule
from core.context import RSPContext
from core.context import RSPSequenceContext
from core.context import add_terms
from core.context import make_context

from .ast import BooleanExpr


_CVC5_DIV_RE = re.compile(r"^\(/ ([^ ]+) ([^ ]+)\)$")
_CVC5_NEG_DIV_RE = re.compile(r"^\(- \(/ ([^ ]+) ([^ ]+)\)\)$")


def _parse_value(value: object) -> float:
    text = str(value)
    if match := _CVC5_DIV_RE.match(text):
        return float(Fraction(match.group(1)) / Fraction(match.group(2)))
    if match := _CVC5_NEG_DIV_RE.match(text):
        return -float(Fraction(match.group(1)) / Fraction(match.group(2)))
    return float(Fraction(text))


@dataclass(frozen=True)
class CEGISVerification:
    result: Verified | Unverified
    counterexample: dict[str, float] | None = None

    @property
    def is_sound(self) -> bool:
        return isinstance(self.result, Verified) or (
            isinstance(self.result, Unverified) and self.result.counterexample is None
        )

    @property
    def is_non_vacuous(self) -> bool:
        return isinstance(self.result, Verified) or (
            isinstance(self.result, Unverified) and self.result.vacuous_certificate is None
        )

    @property
    def is_verified(self) -> bool:
        return isinstance(self.result, Verified)


class CompleteOrderVerifier:
    """cvc5-backed verifier for delay complete-order CEGIS candidates."""

    def __init__(
        self,
        aircraft: tuple[str, ...] = ("p1", "i", "p2", "j", "p3"),
        *,
        integer_arithmetic: bool = False,
        assume_separation_equivalence: bool = True,
    ) -> None:
        self.aircraft = aircraft
        self.integer_arithmetic = integer_arithmetic
        self.assume_separation_equivalence = assume_separation_equivalence

    def symbol_set(self) -> list[str]:
        symbols: list[str] = []
        for label in ("R", "B", "C", "LT", "ET", "LC", "EC"):
            symbols.extend((f"{label}_i", f"{label}_j"))
        for src in self.aircraft:
            for dst in self.aircraft:
                if src in ("i", "j") or dst in ("i", "j"):
                    symbols.append(f"D_{src}_{dst}")
        return symbols

    def verify_rule(self, rule: BooleanExpr) -> CEGISVerification:
        s1, s2, ctx = self._make_problem()
        rule_term = rule.to_cvc5(ctx.solver, self._symbol_table(ctx))
        premises = [*self._base_premises(ctx), rule_term]
        result = verify_pruning_rule(ctx, premises, self._delay_claim(s1, s2))
        counterexample = None
        if isinstance(result, Unverified) and isinstance(result.counterexample, Counterexample):
            counterexample = self._sample_from_counterexample(result.counterexample, ctx)
        return CEGISVerification(result=result, counterexample=counterexample)

    def is_rule_satisfiable(self, rule: BooleanExpr) -> bool:
        _, _, ctx = self._make_problem()
        solver = ctx.solver
        assertions = [
            *ctx.foundational_constraints,
            *self._base_premises(ctx),
            rule.to_cvc5(solver, self._symbol_table(ctx)),
        ]
        solver.push()
        for assertion in assertions:
            solver.assertFormula(assertion)
        result = solver.checkSat()
        solver.pop()
        if result.isUnknown():
            raise RuntimeError(f"Unexpected cvc5 result while checking rule satisfiability: {result}")
        return result.isSat()

    def find_unsound_counterexample(self, rule: BooleanExpr) -> dict[str, float] | None:
        result = self.verify_rule(rule)
        return result.counterexample

    def equivalent(self, left: BooleanExpr, right: BooleanExpr) -> bool:
        solver = cvc5.Solver()
        solver.setLogic("ALL")
        sort = solver.getIntegerSort() if self.integer_arithmetic else solver.getRealSort()
        symbols = {name: solver.mkConst(sort, name) for name in self.symbol_set()}
        left_term = left.to_cvc5(solver, symbols)
        right_term = right.to_cvc5(solver, symbols)
        solver.assertFormula(solver.mkTerm(Kind.NOT, solver.mkTerm(Kind.EQUAL, left_term, right_term)))
        result = solver.checkSat()
        if result.isUnknown():
            return False
        return result.isUnsat()

    def _make_problem(self) -> tuple[RSPSequenceContext, RSPSequenceContext, RSPContext]:
        solver = cvc5.Solver()
        solver.setLogic("ALL")
        solver.setOption("produce-models", "true")
        ctx = make_context(
            self.aircraft,
            solver=solver,
            configure_solver=False,
            integer_arithmetic=self.integer_arithmetic,
        )
        s1 = ctx.with_sequence(self.aircraft)
        s2 = ctx.with_sequence(("p1", "j", "p2", "i", "p3"))
        return s1, s2, ctx

    def _base_premises(self, ctx: RSPContext) -> list[object]:
        if not self.assume_separation_equivalence:
            return []
        return ctx.separation_equivalence("i", "j")

    def _delay_claim(self, s1: RSPSequenceContext, s2: RSPSequenceContext):
        solver = s1.ctx.solver
        return solver.mkTerm(
            Kind.LEQ,
            add_terms(solver, *(s1.delay[aircraft] for aircraft in s1.ctx.aircraft)),
            add_terms(solver, *(s2.delay[aircraft] for aircraft in s2.ctx.aircraft)),
        )

    def _symbol_table(self, ctx: RSPContext) -> dict[str, object]:
        table = {
            "R_i": ctx.r["i"],
            "R_j": ctx.r["j"],
            "B_i": ctx.b["i"],
            "B_j": ctx.b["j"],
            "C_i": ctx.c["i"],
            "C_j": ctx.c["j"],
            "LT_i": ctx.lt["i"],
            "LT_j": ctx.lt["j"],
            "ET_i": ctx.et["i"],
            "ET_j": ctx.et["j"],
            "LC_i": ctx.lc["i"],
            "LC_j": ctx.lc["j"],
            "EC_i": ctx.ec["i"],
            "EC_j": ctx.ec["j"],
        }
        for src in self.aircraft:
            for dst in self.aircraft:
                if src in ("i", "j") or dst in ("i", "j"):
                    table[f"D_{src}_{dst}"] = ctx.delta[(src, dst)]
        return table

    def _sample_from_counterexample(
        self,
        counterexample: Counterexample,
        ctx: RSPContext,
    ) -> dict[str, float]:
        return {
            name: _parse_value(counterexample.value(term))
            for name, term in self._symbol_table(ctx).items()
        }
