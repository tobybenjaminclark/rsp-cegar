from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cvc5 import Kind

from .context import RSPContext


def _is_sat(result) -> bool:
    return result.isSat()


def _is_unsat(result) -> bool:
    return result.isUnsat()


@dataclass(frozen=True)
class RSPSatModel:
    ctx: RSPContext

    def value(self, term) -> str:
        return str(self.ctx.solver.getValue(term))


@dataclass(frozen=True)
class Example(RSPSatModel):
    pass


@dataclass(frozen=True)
class Counterexample(RSPSatModel):
    pass


@dataclass(frozen=True)
class Certificate:
    check: object

    def __post_init__(self) -> None:
        if not _is_unsat(self.check):
            raise ValueError(f"{type(self).__name__} requires an unsat check result.")


@dataclass(frozen=True)
class CorrectnessCertificate(Certificate):
    pass


@dataclass(frozen=True)
class VacuousCertificate(Certificate):
    pass


def _check_with(ctx: RSPContext, assertions: list[object]):
    solver = ctx.solver
    solver.push()
    for assertion in assertions:
        solver.assertFormula(assertion)
    result = solver.checkSat()

    # Important: if sat, the model is only available before pop(). Callers that
    # need model values must keep the solver state alive, so they should not use
    # this helper. It is kept for pure unsat checks.
    solver.pop()
    return result


def check_non_vacuous(ctx: RSPContext, premises: list[object]) -> VacuousCertificate | Example:
    solver = ctx.solver
    solver.push()
    for assertion in list(premises) + ctx.foundational_constraints:
        solver.assertFormula(assertion)
    result = solver.checkSat()
    solver.pop()

    if _is_sat(result):
        return Example(ctx)

    if _is_unsat(result):
        return VacuousCertificate(result)
    raise RuntimeError(f"Unexpected solver result: {result}")


def check_correct(ctx: RSPContext, premises: list[object], claim) -> CorrectnessCertificate | Counterexample:
    solver = ctx.solver
    solver.push()
    for assertion in list(premises) + ctx.foundational_constraints:
        solver.assertFormula(assertion)
    solver.assertFormula(solver.mkTerm(Kind.NOT, claim))
    result = solver.checkSat()

    if _is_sat(result):
        return Counterexample(ctx)

    solver.pop()
    if _is_unsat(result):
        return CorrectnessCertificate(result)
    raise RuntimeError(f"Unexpected solver result: {result}")


@dataclass(frozen=True)
class Verified:
    correctness: CorrectnessCertificate
    example: Example

    @property
    def is_correct(self) -> bool:
        return True

    @property
    def is_non_vacuous(self) -> bool:
        return True

    def __repr__(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class Unverified:
    counterexample: Optional[Counterexample] = None
    vacuous_certificate: Optional[VacuousCertificate] = None

    def __post_init__(self) -> None:
        if self.counterexample is None and self.vacuous_certificate is None:
            raise ValueError("Unverified requires a counterexample, vacuity certificate, or both.")

    @property
    def is_correct(self) -> bool:
        return self.counterexample is None

    @property
    def is_non_vacuous(self) -> bool:
        return self.vacuous_certificate is None

    def __repr__(self) -> str:
        return (
            f"Unverified(counterexample={'yes' if self.counterexample else 'no'}, "
            f"vacuous={'yes' if self.vacuous_certificate else 'no'})"
        )


def verify_pruning_rule(ctx: RSPContext, premises: list[object], claim) -> Verified | Unverified:
    non_vacuity = check_non_vacuous(ctx, premises)
    correctness = check_correct(ctx, premises, claim)

    if isinstance(non_vacuity, Example) and isinstance(correctness, CorrectnessCertificate):
        return Verified(correctness=correctness, example=non_vacuity)

    return Unverified(
        counterexample=correctness if isinstance(correctness, Counterexample) else None,
        vacuous_certificate=non_vacuity if isinstance(non_vacuity, VacuousCertificate) else None,
    )
