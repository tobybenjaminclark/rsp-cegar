from __future__ import annotations

from synthesis.grammar import make_pruning_rule_grammar
from synthesis.synth import log
from synthesis.synth import make_rsp_swap_problem
from synthesis.synth import synthesize_pruning_rule

OBJECTIVE = "delay"
TIMEOUT_MS = 900_000
MAX_CONJUNCTS = 3
REQUIRE_NONVACUOUS = True
SHOW_WITNESS = False


def main() -> None:
    log(f"Configuration: Objective = {OBJECTIVE}, Timeout = {int(TIMEOUT_MS * 0.001):,}s, ")
    problem = make_rsp_swap_problem(timeout_ms=TIMEOUT_MS, objective_name=OBJECTIVE)

    log(f"Symbol Set: [{', '.join(symbol.name.replace('_', '') for symbol in problem.symbols)}]")
    grammar = make_pruning_rule_grammar(problem.env, problem.symbols, MAX_CONJUNCTS)
    result = synthesize_pruning_rule(
        problem,
        grammar=grammar,
        require_nonvacuous=REQUIRE_NONVACUOUS,
    )

    log(f"Non-vacuity witness: {'Synthesized' if REQUIRE_NONVACUOUS else 'Disabled'}")
    if result.rule_solution is not None:
        print(result.rule_solution)
    if SHOW_WITNESS and result.witness_solution is not None:
        print(result.witness_solution)


if __name__ == "__main__":
    main()
