from __future__ import annotations

from cegis.ast import set_symbol_universe
from cegis.cegis import CEGIS
from cegis.normalise import remove_implied_rules
from cegis.seeds import seed_from_synthesis_result
from cegis.verifier import CompleteOrderVerifier
from synthesis.grammar import *
from synthesis.synth import *
from synthesis import *


MAX_ROUNDS = 5
STARTING_POPULATION = 3
GENERATIONS = 3
ELITE = 1
TARGET_SOLUTIONS = 10
SYGUS_TIMEOUT_MS = 10_000
SYGUS_OBJECTIVE = "makespan"


def flat(xs):
    return [y for x in xs for y in (flat(x) if isinstance(x, list) else [x])]


def main() -> int:

    verifier = CompleteOrderVerifier()
    set_symbol_universe(verifier.symbol_set())

    problem = make_rsp_swap_problem(timeout_ms=SYGUS_TIMEOUT_MS, objective_name=SYGUS_OBJECTIVE)

    # Construct the grammar.
    conj = NonTerminal("Rule", sort=problem.env.bool_sort)
    cmp = NonTerminal("Atom", sort=problem.env.bool_sort)

    by_name = {symbol.name: symbol for symbol in problem.symbols}
    names = set(by_name)

    prefixes = sorted({
        name[:-2]
        for name in names
        if name.endswith("_i") and not name.startswith(("D_", "CTOT", "DELAY")) and name[:-2] != "T"
    })
    comparable_pairs = [
        *[(f"{p}_i", f"{p}_j") for p in prefixes if f"{p}_j" in names],
        *[
            pair
            for pair in (("D_i_x", "D_j_x"), ("D_x_i", "D_x_j"))
            if pair[0] in names and pair[1] in names
        ],
    ]

    grammar = Grammar(
        nonterminals=(conj, cmp),
        terminals=problem.symbols,
        start=conj,
        productions=(
            conj >> (cmp | (cmp & cmp) | (cmp & cmp & cmp) | (cmp & cmp & cmp & cmp)),
            cmp >> Choice(tuple(flat(
                [[by_name[l] <= by_name[r], by_name[r] <= by_name[l], by_name[l].eq(by_name[r])]
                 for l, r in comparable_pairs]
            ))),
        ),
    )
    log(f"Visualising Context-Free Grammar for SyGuS:\n\n{grammar.vis()}\n")

    # Run Synthesis
    result = synthesize_pruning_rule(
        problem,
        grammar=grammar,
        require_nonvacuous=True,
    )

    seed_rules = [seed_from_synthesis_result(verifier, result)]

    cegis = CEGIS(
        verifier,
        max_rounds=MAX_ROUNDS,
        starting=STARTING_POPULATION,
        generations=GENERATIONS,
        elite=ELITE,
        target_solutions=TARGET_SOLUTIONS,
        seed_rules=seed_rules,
    )
    rules = cegis.synthesise()

    if not rules:
        log("No Pruning Rules found.")
        return 0

    log("Initial Verified Pruning Ruleset:\n")
    for rule in rules:
        print(f" ► {rule}")

    log("Normalising Discovered Ruleset")
    rules = remove_implied_rules(rules, verifier)
    for rule in rules:
        print(f" ► {rule}")


if __name__ == "__main__":
    raise SystemExit(main())
