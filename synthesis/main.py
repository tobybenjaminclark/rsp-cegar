from __future__ import annotations

from synthesis.grammar import Choice
from synthesis.grammar import Grammar
from synthesis.grammar import NonTerminal
from synthesis.synth import log
from synthesis.synth import make_rsp_swap_problem
from synthesis.synth import synthesize_pruning_rule

OBJECTIVE = "delay+ctot"
TIMEOUT_MS = 900_000
REQUIRE_NONVACUOUS = True
SHOW_WITNESS = False


def main() -> None:
    problem = make_rsp_swap_problem(timeout_ms=10000, objective_name='delay')

    log(f"Symbol Set: [{', '.join(symbol.name.replace('_', '') for symbol in problem.symbols)}]")

    # Construct the grammar.
    conj = NonTerminal("Rule", sort=problem.env.bool_sort)
    cmp = NonTerminal("Atom", sort=problem.env.bool_sort)

    by_name = {symbol.name: symbol for symbol in problem.symbols}

    flat = lambda xs: [y for x in xs for y in (flat(x) if isinstance(x, list) else [x])]
    names = set(by_name)

    prefixes = sorted({name[:-2] for name in names if name.endswith("_i") and not name.startswith(("D_", "T", "DELAY", "CTOT")) and name[:-2] != "T"})
    comparable_pairs = [
        *[(f"{p}_i", f"{p}_j") for p in prefixes if f"{p}_j" in names],
        *[pair for pair in (("D_i_x", "D_j_x"), ("D_x_i", "D_x_j")) if pair[0] in names and pair[1] in names],
    ]

    grammar = Grammar(
        nonterminals=(conj, cmp),
        terminals=problem.symbols,
        start=conj,
        productions=(
            conj >> (cmp | (cmp & cmp) | (cmp & cmp & cmp)),
            cmp >> Choice(tuple(flat(
                [[by_name[l] <= by_name[r], by_name[r] <= by_name[l], by_name[l].eq(by_name[r])] for l, r in comparable_pairs]
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


if __name__ == "__main__":
    main()
