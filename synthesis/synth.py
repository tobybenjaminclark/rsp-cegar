from __future__ import annotations

import datetime
from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Sequence

import cvc5
from cvc5 import Kind

from core.context import RSPContext
from core.context import RSPSequenceContext
from core.context import add_terms
from core.context import make_context
from synthesis.grammar import Terminal
from synthesis.symbols import make_allowed_symbols


def and_terms(solver: cvc5.Solver, *terms):
    if not terms:           return solver.mkTrue()
    elif len(terms) == 1:   return terms[0]
    return                  solver.mkTerm(Kind.AND, *terms)


def or_terms(solver: cvc5.Solver, *terms):
    if not terms:           return solver.mkFalse()
    elif len(terms) == 1:   return terms[0]
    return                  solver.mkTerm(Kind.OR, *terms)


def log(message: str) -> None:
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def define_fun_to_string(f: object, params: Iterable[object], body: object) -> str:
    return (
        f"(define-fun {f} ("
        f"{reduce(lambda acc, p: acc + (' ' if acc else '') + f'({p} {p.getSort()})', params, '')}"
        f") "
        f"{f.getSort().getFunctionCodomainSort() if f.getSort().isFunction() else f.getSort()} "
        f"{body})"
    )


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


@dataclass(frozen=True)
class WitnessSymbol:
    name: str
    actual: object


@dataclass(frozen=True)
class ObjectiveComponent:
    name: str
    left: object
    right: object

    def claim(self, env: SygusEnv):
        return env.solver.mkTerm(Kind.LEQ, self.left, self.right)


@dataclass(frozen=True)
class SygusProblem:
    env: SygusEnv
    ctx: RSPContext
    objective: ObjectiveComponent
    symbols: tuple[Terminal, ...]
    background_constraints: tuple[object, ...] = ()
    witness_symbols: tuple[WitnessSymbol, ...] = ()
    seq_ij: tuple[str, ...] | None = None
    seq_ji: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SynthesisResult:
    problem: SygusProblem
    rule: object
    witnesses: tuple[object, ...]
    check: object
    rule_solution: str | None
    witness_solution: str | None


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
    symbols = make_allowed_symbols(ctx, s_ij, s_ji)
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


def apply_rule(env: SygusEnv, rule, args: list[object]):
    return env.solver.mkTerm(Kind.APPLY_UF, rule, *args)


def symbol_actual_for_aircraft(problem: SygusProblem, symbol: Terminal, aircraft: str):
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


def rule_args_for_aircraft(
    problem: SygusProblem,
    aircraft: str,
    substitutions: tuple[tuple[object, object], ...] = (),
) -> list[object]:
    return [
        substitute_all(symbol_actual_for_aircraft(problem, symbol, aircraft), substitutions)
        for symbol in problem.symbols
    ]


def rule_instances(
    problem: SygusProblem,
    rule,
    substitutions: tuple[tuple[object, object], ...] = (),
) -> list[object]:
    return [
        apply_rule(problem.env, rule, rule_args_for_aircraft(problem, aircraft, substitutions))
        for aircraft in problem.ctx.aircraft
    ]


def synthesize_witnesses(problem: SygusProblem) -> tuple[object, ...]:
    solver = problem.env.solver
    witness_sort = problem.ctx.real_sort
    return tuple(
        solver.synthFun(f"w_{symbol.name}", [], witness_sort)
        for symbol in problem.witness_symbols
    )


def substitute_all(term, substitutions: tuple[tuple[object, object], ...]):
    current = term
    for old, new in substitutions:
        current = current.substitute(old, new)
    return current


def witness_substitutions(
    witness_symbols: tuple[WitnessSymbol, ...],
    witnesses: tuple[object, ...],
) -> tuple[tuple[object, object], ...]:
    return tuple(
        (symbol.actual, witness)
        for symbol, witness in zip(witness_symbols, witnesses)
    )


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


def synthesize_pruning_rule(
    problem: SygusProblem,
    grammar,
    require_nonvacuous: bool = True,
) -> SynthesisResult:
    env = problem.env
    solver = env.solver
    symbols = problem.symbols

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
