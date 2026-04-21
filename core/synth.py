from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Sequence
import os
import sys
import cvc5
from cvc5 import Kind

OBJECTIVE = "delay"
TIMEOUT_MS = 900_000
MAX_CONJUNCTS = 3
REQUIRE_NONVACUOUS = True
SHOW_WITNESS = False

# Allow direct execution: `python core/synth.py`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.context import RSPContext
from core.context import RSPSequenceContext
from core.context import add_terms
from core.context import make_context


# Function to apply conjunction operators over a set of terms.
def and_terms(solver: cvc5.Solver, *terms):
    if not terms:           return solver.mkTrue()
    elif len(terms) == 1:   return terms[0]
    else:                   return solver.mkTerm(Kind.AND, *terms)


# Function to apply disjunction operators over a set of terms.
def or_terms(solver: cvc5.Solver, *terms):
    if not terms:           return solver.mkFalse()
    elif len(terms) == 1:   return terms[0]
    else:                   return solver.mkTerm(Kind.OR, *terms)


# Function to define a simple information log.
def log(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# Function to convert a CVC5 function to its SMTLIB2 string definition.
def define_fun_to_string(f: object, params: Iterable[object], body: object) -> str:
    return (
        f"(define-fun {f} ("
        f"{reduce(lambda acc, p: acc + (' ' if acc else '') + f'({p} {p.getSort()})', params, '')}"
        f") "
        f"{f.getSort().getFunctionCodomainSort() if f.getSort().isFunction() else f.getSort()} "
        f"{body})"
    )


# Function to convert a set of solutions to an SMTLIB2 string definition.
def synth_solutions_to_string(terms: Sequence[object], sols: Sequence[object]) -> str:
    parts = lambda i: (
        sols[i][0] if sols[i].getKind() == Kind.LAMBDA else [],
        sols[i][1] if sols[i].getKind() == Kind.LAMBDA else sols[i],
    )
    return (
        "(\n"
        + "".join(
            f"  {define_fun_to_string(term, parts(index)[0], parts(index)[1])}\n"
            for index, term in enumerate(terms)
        )
        + ")"
    )


# Class to build and configure the SyGuS CVC5 environment.
class SygusEnv:
    def __init__(self, logic: str = "LIA", timeout_ms: int = 10_000, integer_arithmetic: bool = True) -> None:
        self.integer_arithmetic = integer_arithmetic
        self.solver = cvc5.Solver()
        self.solver.setOption("sygus", "true")
        self.solver.setOption("incremental", "false")
        if timeout_ms > 0:
            self.solver.setOption("tlimit", str(timeout_ms))
            self.solver.setOption("tlimit-per", str(timeout_ms))
            self.solver.setOption("rlimit", str(timeout_ms * 1000))
        self.solver.setLogic(logic)
        self.real_sort = self.solver.getIntegerSort() if integer_arithmetic else self.solver.getRealSort()
        self.bool_sort = self.solver.getBooleanSort()

    def real(self, name: str):
        return self.solver.declareSygusVar(name, self.real_sort)

    def real_val(self, value: int | str):
        if self.integer_arithmetic:
            return self.solver.mkInteger(int(value))
        return self.solver.mkReal(str(value))


# Dataclass to represent a symbolic grammar variable and its concrete interpretation.
@dataclass(frozen=True)
class SygusSymbol:
    name: str
    formal: object
    actual: object | None


# Dataclass to represent a witness symbol used for non-vacuity constraints.
@dataclass(frozen=True)
class WitnessSymbol:
    name: str
    actual: object


# Dataclass to represent an objective comparison between two sequence terms.
@dataclass(frozen=True)
class ObjectiveComponent:
    name: str
    left: object
    right: object

    def claim(self, env: SygusEnv):
        return env.solver.mkTerm(Kind.LEQ, self.left, self.right)


# Dataclass to represent the complete synthesis problem instance.
@dataclass(frozen=True)
class SygusProblem:
    env: SygusEnv
    ctx: RSPContext
    objective: ObjectiveComponent
    symbols: tuple[SygusSymbol, ...]
    background_constraints: tuple[object, ...] = ()
    witness_symbols: tuple[WitnessSymbol, ...] = ()
    seq_ij: tuple[str, ...] | None = None
    seq_ji: tuple[str, ...] | None = None


# Dataclass to represent the result of the synthesis query.
@dataclass(frozen=True)
class SynthesisResult:
    problem: SygusProblem
    rule: object
    witnesses: tuple[object, ...]
    check: object
    rule_solution: str | None
    witness_solution: str | None


# Function to build the base RSP swap synthesis problem.
def make_rsp_swap_problem(
    timeout_ms: int = 10_000,
    objective_name: str = "makespan",
) -> SygusProblem:
    seq_ij = ("a", "i", "b", "j", "c")
    seq_ji = ("a", "j", "b", "i", "c")
    env = SygusEnv(timeout_ms=timeout_ms)
    aircraft = tuple(dict.fromkeys(seq_ij))
    ctx = make_context(
        aircraft,
        env.solver,
        configure_solver=False,
        integer_arithmetic=env.integer_arithmetic,
        use_sygus_vars=True,
    )
    s_ij = ctx.with_sequence(seq_ij)
    s_ji = ctx.with_sequence(seq_ji)

    objective = make_rsp_objective(s_ij, s_ji, objective_name)
    symbols = make_allowed_symbols(ctx)
    witness_symbols = make_context_witness_symbols(ctx)
    background_constraints = tuple(ctx.foundational_constraints)
    return SygusProblem(
        env=env,
        ctx=ctx,
        objective=objective,
        symbols=symbols,
        background_constraints=background_constraints,
        witness_symbols=witness_symbols,
        seq_ij=s_ij.seq,
        seq_ji=s_ji.seq,
    )


# Function to derive the objective term pair for the selected objective.
def make_rsp_objective(
    s_ij: RSPSequenceContext,
    s_ji: RSPSequenceContext,
    objective_name: str,
) -> ObjectiveComponent:
    solver = s_ij.ctx.solver

    if objective_name == "makespan":
        last_plane = s_ij.seq[-1]
        return ObjectiveComponent(
            name=f"last-takeoff makespan component T_{last_plane}",
            left=s_ij.takeoff[last_plane],
            right=s_ji.takeoff[last_plane],
        )

    if objective_name == "delay":
        return ObjectiveComponent(
            name="total delay",
            left=add_terms(solver, *(s_ij.delay[plane] for plane in s_ij.ctx.aircraft)),
            right=add_terms(solver, *(s_ji.delay[plane] for plane in s_ij.ctx.aircraft)),
        )

    raise ValueError(f"Unknown objective: {objective_name}")


# Function to collect all context symbols that may appear in witness substitutions.
def make_context_witness_symbols(ctx: RSPContext) -> tuple[WitnessSymbol, ...]:
    symbols: list[WitnessSymbol] = []
    seen: set[str] = set()

    def add_symbol(name: str, actual) -> None:
        if name in seen:
            return
        seen.add(name)
        symbols.append(WitnessSymbol(name, actual))

    for label, mapping in (
        ("r", ctx.r),
        ("b", ctx.b),
        ("c", ctx.c),
        ("ec", ctx.ec),
        ("lc", ctx.lc),
        ("et", ctx.et),
        ("lt", ctx.lt),
    ):
        for plane in ctx.aircraft:
            add_symbol(f"{label}_{plane}", mapping[plane])

    for src in ctx.aircraft:
        for dst in ctx.aircraft:
            add_symbol(f"d_{src}_{dst}", ctx.delta[(src, dst)])

    return tuple(symbols)


# Function to collect the allowed schema symbols for pruning-rule synthesis.
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


# Function to map symbol names to their formal grammar terms.
def symbol_map(symbols: tuple[SygusSymbol, ...]) -> dict[str, object]:
    return {symbol.name: symbol.formal for symbol in symbols}


# Function to define the allowed atomic predicates in the grammar.
def allowed_predicates(env: SygusEnv, symbols: tuple[SygusSymbol, ...]) -> list[object]:
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


# Function to build the pruning rule grammar with bounded conjunction width.
def make_pruning_rule_grammar(env: SygusEnv, symbols: tuple[SygusSymbol, ...], max_conjuncts: int):
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


# Function to apply the synthesized pruning rule over a concrete argument list.
def apply_rule(env: SygusEnv, rule, args: list[object]):
    return env.solver.mkTerm(Kind.APPLY_UF, rule, *args)


# Function to instantiate schema symbols for a specific aircraft.
def symbol_actual_for_aircraft(problem: SygusProblem, symbol: SygusSymbol, aircraft: str):
    if symbol.name == "D_i_x":
        return problem.ctx.delta[("i", aircraft)]
    if symbol.name == "D_j_x":
        return problem.ctx.delta[("j", aircraft)]
    if symbol.name == "D_x_i":
        return problem.ctx.delta[(aircraft, "i")]
    if symbol.name == "D_x_j":
        return problem.ctx.delta[(aircraft, "j")]
    if symbol.actual is None:
        raise ValueError(f"No concrete interpretation for schema symbol {symbol.name}")
    return symbol.actual


# Function to build the rule-argument vector for one aircraft.
def rule_args_for_aircraft(
    problem: SygusProblem,
    aircraft: str,
    substitutions: tuple[tuple[object, object], ...] = (),
) -> list[object]:
    return [
        substitute_all(symbol_actual_for_aircraft(problem, symbol, aircraft), substitutions)
        for symbol in problem.symbols
    ]


# Function to instantiate the synthesized rule over all aircraft in the context.
def rule_instances(
    problem: SygusProblem,
    rule,
    substitutions: tuple[tuple[object, object], ...] = (),
) -> list[object]:
    return [
        apply_rule(problem.env, rule, rule_args_for_aircraft(problem, aircraft, substitutions))
        for aircraft in problem.ctx.aircraft
    ]


# Function to declare synthesized witness constants used for non-vacuity.
def synthesize_witnesses(problem: SygusProblem) -> tuple[object, ...]:
    solver = problem.env.solver
    witness_sort = problem.ctx.real_sort
    return tuple(
        solver.synthFun(f"w_{symbol.name}", [], witness_sort)
        for symbol in problem.witness_symbols
    )


# Function to apply a sequence of substitutions over a term.
def substitute_all(term, substitutions: tuple[tuple[object, object], ...]):
    current = term
    for old, new in substitutions:
        current = current.substitute(old, new)
    return current


# Function to pair each witness symbol with its synthesized witness term.
def witness_substitutions(
    witness_symbols: tuple[WitnessSymbol, ...],
    witnesses: tuple[object, ...],
) -> tuple[tuple[object, object], ...]:
    return tuple(
        (symbol.actual, witness)
        for symbol, witness in zip(witness_symbols, witnesses)
    )


# Function to add the non-vacuity synthesis constraint.
def add_nonvacuity_constraint(problem: SygusProblem, rule, witnesses: tuple[object, ...]) -> None:
    env = problem.env
    solver = env.solver
    substitutions = witness_substitutions(problem.witness_symbols, witnesses)
    witness_background = [
        substitute_all(constraint, substitutions)
        for constraint in problem.background_constraints
    ]
    witness_rule_instances = rule_instances(problem, rule, substitutions)

    nonvacuity = and_terms(
        solver,
        and_terms(solver, *witness_background),
        and_terms(solver, *witness_rule_instances),
    )
    solver.addSygusConstraint(nonvacuity)


# Function to synthesize a pruning rule under safety and optional non-vacuity constraints.
def synthesize_pruning_rule(problem: SygusProblem, require_nonvacuous: bool = True, max_conjuncts: int = 5) -> SynthesisResult:
    env = problem.env
    solver = env.solver
    symbols = problem.symbols

    grammar = make_pruning_rule_grammar(env, symbols, max_conjuncts)
    rule = solver.synthFun("prune", [symbol.formal for symbol in symbols], env.bool_sort, grammar)
    witnesses = synthesize_witnesses(problem) if require_nonvacuous else ()

    rule_on_rsp_vars = and_terms(solver, *rule_instances(problem, rule))
    valid_rsp = and_terms(solver, *problem.background_constraints)
    safety = or_terms(
        solver,
        solver.mkTerm(Kind.NOT, and_terms(solver, valid_rsp, rule_on_rsp_vars)),
        problem.objective.claim(env),
    )
    solver.addSygusConstraint(safety)

    if require_nonvacuous:
        add_nonvacuity_constraint(problem, rule, witnesses)

    log("Invoking Synthesis (This may take some time)")
    check = solver.checkSynth()
    log(f"Synthesis Complete, Result: {check}")
    if not check.hasSolution():
        return SynthesisResult(problem, rule, witnesses, check, None, None)

    rule_solution = synth_solutions_to_string([rule], solver.getSynthSolutions([rule]))
    witness_solution = None
    if witnesses:
        witness_terms = list(witnesses)
        witness_solution = synth_solutions_to_string(witness_terms, solver.getSynthSolutions(witness_terms))
    return SynthesisResult(problem, rule, witnesses, check, rule_solution, witness_solution)


# Function to run synthesis with top-level configuration constants.
def main() -> None:
    log(f"Configuration: Objective = {OBJECTIVE}, Timeout = {int(TIMEOUT_MS * 0.001):,}s, ")
    problem = make_rsp_swap_problem(timeout_ms=TIMEOUT_MS, objective_name=OBJECTIVE)

    log(f"Symbol Set: [{', '.join(symbol.name.replace("_", "") for symbol in problem.symbols)}]")
    result = synthesize_pruning_rule(
        problem,
        require_nonvacuous=REQUIRE_NONVACUOUS,
        max_conjuncts=MAX_CONJUNCTS,
    )

    log(f"Non-vacuity witness: {'Synthesized' if REQUIRE_NONVACUOUS else 'Disabled'}")
    if result.rule_solution is not None:
        print(result.rule_solution)
    if SHOW_WITNESS and result.witness_solution is not None:
        print(result.witness_solution)


if __name__ == "__main__":
    main()
