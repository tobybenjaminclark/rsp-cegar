from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from itertools import product
from enum import Enum
from itertools import chain
from functools import reduce
import z3
from .pr_ast import BooleanExpr


# Symbolic Maximums in Z3
def zmax(x, y):     return z3.If(x >= y, x, y)
def zmax_list(xs):  return reduce(lambda a, b: zmax(a, b), xs)





class Type(Enum):
    Integer = "Int"
    Real    = "Real"

    def mk(self, name): return z3.Int(name) if self is Type.Integer else z3.Real(name)





@dataclass
class Form:
    variables:   List[Tuple[str, Type]] = field(default_factory=list)
    attributes:  List[Tuple[str, Type]] = field(default_factory=list)
    constraints: List = field(default_factory=list)

    def instantiate(self):
        Σ = {}
        for n, t in self.variables:     Σ[n] = t.mk(n)
        for n, t in self.attributes:    Σ[n] = t.mk(n)
        self.Σ = Σ

    def instantiate_constraints(self, Σ):       return [c(Σ) for c in self.constraints]
    def symbol_set(self):                       return [n for (n, _) in self.variables + self.attributes]
    def verify_rule(self):                      raise NotImplementedError()






class CompleteOrderForm(Form):
    def __init__(self):
        super().__init__()

        self.aircraft = ["σ1", "i", "σ2", "j", "σ3"]

        # Type declarations
        self.variables = (
            [(f"R_{ac}", Type.Real) for ac in self.aircraft] +
            [(f"B_{ac}", Type.Real) for ac in self.aircraft] +
            [(f"C_{ac}", Type.Real) for ac in self.aircraft] +
            [(f"LT_{ac}", Type.Real) for ac in self.aircraft] +
            [(f"ET_{ac}", Type.Real) for ac in self.aircraft] +
            [(f"LC_{ac}", Type.Real) for ac in self.aircraft] +
            [(f"EC_{ac}", Type.Real) for ac in self.aircraft]
        )

        # δ(x,y) for all pairs
        self.attributes = [(f"δ_{x}_{y}", Type.Real)
                           for x, y in product(self.aircraft, self.aircraft)]

        self.instantiate()

        def axiom_nonnegative(Σ = self.Σ):          return z3.And(*[Σ[f"B_{ac}"]  >= 0 for ac in self.aircraft] + [Σ[f"LT_{ac}"] >= 0 for ac in self.aircraft] + [Σ[f"R_{ac}"] >= 0 for ac in self.aircraft])
        def axiom_r_after_b(Σ = self.Σ):            return z3.And(*[Σ[f"R_{ac}"] >= Σ[f"B_{ac}"] for ac in self.aircraft])
        def axiom_delta_bounds(Σ = self.Σ):         return z3.And(*[z3.And(Σ[f"δ_{x}_{y}"] >= 0, Σ[f"δ_{x}_{y}"] < 5) for x, y in product(self.aircraft, self.aircraft)])
        def axiom_ab_positive_equal(Σ = self.Σ):    return z3.And(Σ["δ_i_j"] > 0, Σ["δ_j_i"] > 0, Σ["δ_i_j"] == Σ["δ_j_i"])

        def axiom_ab_identical(Σ = self.Σ):
            pairs = map(lambda x: [
                Σ[f"δ_i_{x}"] == Σ[f"δ_j_{x}"],
                Σ[f"δ_{x}_i"] == Σ[f"δ_{x}_j"]
            ], [x for x in self.aircraft if x not in ("i", "j")])
            return z3.And(*chain.from_iterable(pairs))

        self.constraints = [
            axiom_nonnegative,
            axiom_r_after_b,
            axiom_delta_bounds,
            axiom_ab_positive_equal,
            axiom_ab_identical,
        ]

    def symbol_set(self):
        # keep only symbols involving i or j
        good = []
        for (n, _) in self.variables + self.attributes:
            if "_i" in n or "_j" in n:  # only names referencing i or j
                good.append(n)
        return good

    def compute_T(self, _seq):
        T, Σ = {}, self.Σ
        T[_seq[0]] = Σ[f"C_{_seq[0]}"]

        for i in range(1, len(_seq)):
            preds = _seq[:i]
            T[_seq[i]] = zmax(
                Σ[f"R_{_seq[i]}"],
                zmax_list([T[x] + Σ[f"δ_{x}_{_seq[i]}"] for x in preds])
            )
        return T

    def delay_cost(self, T):  return z3.Sum(([((T[x] - self.Σ[f"C_{x}"])) for x in self.aircraft]))

    def verify_rule(self, rule):
        SEQ_1 = ["σ1", "i", "σ2", "j", "σ3"]
        SEQ_2 = ["σ1", "j", "σ2", "i", "σ3"]
        T1 = self.compute_T(SEQ_1)
        T2 = self.compute_T(SEQ_2)
        D1 = self.delay_cost(T1)
        D2 = self.delay_cost(T2)

        # Check SAT
        solver_vacuous = z3.Solver()
        for c in self.constraints: solver_vacuous.add(c())
        solver_vacuous.add(rule.to_z3())

        solver_vacuous.add(D1 <= D2)
        if solver_vacuous.check() != z3.sat:
            print(f"[SMT] : Rule '{rule}' holds vacuously unsatisfiable.")
            return False

        elif solver_vacuous.check() == z3.unknown:
            print(f"[SMT] : Rule '{rule}' cannot be vacuous-verified in SMT-logic")
            return False

        solver_proof = z3.Solver()
        for c in self.constraints: solver_proof.add(c())
        solver_proof.add(rule.to_z3())

        solver_proof.add(D1 > D2)
        if solver_proof.check() == z3.sat:
            print(f"[SMT] : Rule '{rule}' is not sound (counterexample found).")
            return False

        elif solver_proof.check() == z3.unknown:
            print(f"[SMT] : Rule '{rule}' cannot be soundness-verified in SMT-logic")
            return False

        elif solver_proof.check() == z3.unsat:
            print(f"[SMT] : Rule '{rule}' is sound (no counterexample found) and non-degenerate.")
            return True

    def find_unsound_counterexample(self, rule: BooleanExpr) -> z3.Model:
        SEQ_1 = ["σ1", "i", "σ2", "j", "σ3"]
        SEQ_2 = ["σ1", "j", "σ2", "i", "σ3"]
        T1 = self.compute_T(SEQ_1)
        T2 = self.compute_T(SEQ_2)
        D1 = self.delay_cost(T1)
        D2 = self.delay_cost(T2)

        s = z3.Solver()
        for c in self.constraints:
            s.add(c())
        s.add(rule.to_z3())
        s.add(D1 > D2)

        if s.check() == z3.sat:
            return s.model()
        else:
            return None

    def is_rule_satisfiable(self, rule: BooleanExpr) -> bool:
        """Check ∃ model. constraints ∧ rule."""
        s = z3.Solver()
        for c in self.constraints:
            s.add(c())
        s.add(rule.to_z3())
        return s.check() == z3.sat