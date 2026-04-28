from __future__ import annotations

from synthesis.grammar import Choice
from synthesis.grammar import Grammar
from synthesis.grammar import NonTerminal
from synthesis.grammar import set_smt_env
from synthesis.synth import log
from synthesis.synth import make_rsp_swap_problem
from synthesis.synth import synthesize_pruning_rule

OBJECTIVE = "delay"
TIMEOUT_MS = 900_000
REQUIRE_NONVACUOUS = True
SHOW_WITNESS = False


def main() -> None:

    log(f"Configuration: Objective = {OBJECTIVE}, Timeout = {int(TIMEOUT_MS * 0.001):,}s, ")
    problem = make_rsp_swap_problem(timeout_ms=TIMEOUT_MS, objective_name=OBJECTIVE)

    log(f"Symbol Set: [{', '.join(symbol.name.replace('_', '') for symbol in problem.symbols)}]")

    #
    # Construct the grammar.
    #
    conj = NonTerminal("Rule", sort = problem.env.bool_sort)
    cmp = NonTerminal("Atom", sort = problem.env.bool_sort)
    aexp = NonTerminal("AExpr", sort = problem.env.real_sort)
    dt = NonTerminal("Dt", sort = problem.env.real_sort)

    by_name = {symbol.name: symbol for symbol in problem.symbols}
    objective_term_names = (
        "T_i",
        "T_j",
        "T'_i",
        "T'_j",
        "B_i",
        "B_j",
    )
    objective_terms = tuple(by_name[name] for name in objective_term_names if name in by_name)

    flat = lambda xs: [y for x in xs for y in (flat(x) if isinstance(x, list) else [x])]
    names = set(by_name)

    prefixes = sorted({
        name[:-2]
        for name in names
        if name.endswith("_i") and not name.startswith("D_") and not name.startswith("T")
    })
    comparable_pairs = [
        *[(f"{p}_i", f"{p}_j") for p in prefixes if f"{p}_j" in names],
        *[pair for pair in (("D_i_x", "D_j_x"), ("D_x_i", "D_x_j")) if pair[0] in names and pair[1] in names],
    ]

    grammar = Grammar(
        nonterminals=(conj, cmp, aexp, dt),
        terminals=problem.symbols,
        start=conj,
        productions=(
            conj >> (cmp | (cmp & cmp) | (cmp & cmp & cmp)),
            cmp >> (Choice(tuple(flat(
                [[by_name[l] <= by_name[r], by_name[r] <= by_name[l], by_name[l].eq(by_name[r])] for l, r in comparable_pairs]
            ))) | (aexp <= aexp) | aexp.eq(aexp)),
            aexp >> (Choice(objective_terms) | dt | (aexp + aexp) | (aexp - aexp)),
            dt >> (
                (by_name["T_i"] - by_name["B_i"])
                | (by_name["T'_i"] - by_name["B_i"])
                | (by_name["T_j"] - by_name["B_j"])
                | (by_name["T'_j"] - by_name["B_j"])
            ),
        ),
    )

    #
    # Binds Terminal symbols to problem variables.
    #
    set_smt_env(
        symbol_table={symbol.name: symbol.formal for symbol in problem.symbols},
    )

    grammar = grammar.to_cvc5(problem.env.solver)


    #
    # Run Synthesis
    #
    result = synthesize_pruning_rule(
        problem,
        grammar=grammar,
        require_nonvacuous=REQUIRE_NONVACUOUS,
    )


if __name__ == "__main__":
    main()
