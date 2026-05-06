from __future__ import annotations

import math

from cegis.ast import set_symbol_universe
from cegis.cegis import CEGIS
from cegis.verifier import CompleteOrderVerifier


MAX_ROUNDS = 100
STARTING_POPULATION = 10
GENERATIONS = 30
ELITE = 5
TARGET_SOLUTIONS = math.inf


def main() -> int:
    if ELITE > STARTING_POPULATION:
        raise ValueError("ELITE cannot be greater than STARTING_POPULATION.")

    verifier = CompleteOrderVerifier()
    set_symbol_universe(verifier.symbol_set())

    cegis = CEGIS(
        verifier,
        max_rounds=MAX_ROUNDS,
        starting=STARTING_POPULATION,
        generations=GENERATIONS,
        elite=ELITE,
        target_solutions=TARGET_SOLUTIONS,
    )
    rules = cegis.synthesise()

    if not rules:
        print("\nNo Pruning Rules found.")
        return 0

    print("\nGenerated & Verified Pruning Conditions:")
    for rule in rules:
        print(f"\t*\t{rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
