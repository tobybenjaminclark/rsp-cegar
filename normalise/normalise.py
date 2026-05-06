from __future__ import annotations
from collections.abc import Iterable
from cvc5 import Kind
from weakening.ast import BooleanExpr
from weakening.verifier import CompleteOrderVerifier



def implies(antecedent: BooleanExpr, consequent: BooleanExpr, verifier: CompleteOrderVerifier) -> bool:
    """Return true when antecedent semantically implies consequent."""

    _, _, ctx = verifier._make_problem()
    solver = ctx.solver
    symbols = verifier._symbol_table(ctx)
    solver.push()
    for assertion in ctx.foundational_constraints:
        solver.assertFormula(assertion)
    solver.assertFormula(antecedent.to_cvc5(solver, symbols))
    solver.assertFormula(solver.mkTerm(Kind.NOT, consequent.to_cvc5(solver, symbols)))
    result = solver.checkSat()
    solver.pop()
    if result.isUnknown():
        return False
    return result.isUnsat()



def remove_implied_rules(rules: Iterable[BooleanExpr],verifier: CompleteOrderVerifier) -> list[BooleanExpr]:
    """Remove any rule that is implied by another rule."""

    unique_rules = _deduplicate(rules)
    return [
        candidate
        for candidate in unique_rules
        if not any(
            str(candidate) != str(other) and implies(candidate, other, verifier)
            for other in unique_rules
        )
    ]



def _deduplicate(rules: Iterable[BooleanExpr]) -> list[BooleanExpr]:
    kept = []
    seen = set()
    for rule in rules:
        key = str(rule)
        if key not in seen:
            kept.append(rule)
            seen.add(key)
    return kept
