from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

from rsp_cegis.cegis import CEGIS
from rsp_cegis.form import CompleteOrderForm
from rsp_cegis.pr_ast import set_symbol_universe

VERSION = "v1.0.0"
PROMPT_WIDTH = 70



# Define a helper function to parse a positive integer.
def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected integer, got: {value!r}") from e
    if n <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got: {n}")
    return n



# Define a helper function to parse a non-negative integer
def _nonnegative_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected integer, got: {value!r}") from e
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got: {n}")
    return n



# Define a helper function to parse target solutions (integer | ∞)
def _target_solutions(value: str) -> float:
    v = value.strip().lower()
    if v in {"inf", "infty", "infinite", "infinity", "math.inf"}:
        return math.inf
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "expected an integer number of solutions or 'inf'"
        ) from e
    if n <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got: {n}")
    return float(n)



# Define a helper function to prompt for the limit of target solutions.
def prompt_target_line(label: str, *, default):
    while True:
        raw = input(f"{label.ljust(PROMPT_WIDTH)}: ").strip().lower()
        if raw == "":
            return default
        try:
            return _target_solutions(raw)
        except argparse.ArgumentTypeError as e:
            print(f"  {e}")



# Define a helper function to prompt for attributes (polymorphic)
def prompt_line(label: str, *, default=None, min_value=None):
    while True:
        raw = input(f"{label.ljust(PROMPT_WIDTH)}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            n = int(raw)
        except ValueError:
            print("  Please enter an integer.")
            continue
        if min_value is not None and n < min_value:
            print(f"  Value must be ≥ {min_value}.")
            continue
        return n





# Define a function to construct the `argparse` argument parser.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rsp-cegis",
        description="Synthesize runway-sequencing pruning rules using CEGIS.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="show version and exit",
    )

    p.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="prompt for parameters on startup",
    )

    p.add_argument("--max-rounds", type=_positive_int, default=100,
                   help="CEGIS outer rounds")

    p.add_argument("--starting", type=_positive_int, default=10,
                   help="initial population size")

    p.add_argument("--generations", type=_positive_int, default=30,
                   help="GA generations per round")

    p.add_argument("--elite", type=_positive_int, default=5,
                   help="elite count")

    p.add_argument("--target-solutions", type=_target_solutions, default=math.inf,
                   help="stop after N solutions (or 'inf')")

    return p





# Define a general, high-level run function (dependent on arguments)
def run(args: argparse.Namespace) -> int:
    form = CompleteOrderForm()
    set_symbol_universe(form.symbol_set())

    # If the program is run interactively, gather arguments in the CLI.
    if args.interactive:
        print("Interactive mode — configure the CEGIS search\n")

        args.max_rounds = prompt_line(
            "Enter the number of outer CEGIS iterations / rounds",
            default=args.max_rounds,
            min_value=1,
        )

        args.starting = prompt_line(
            "Enter the initial genetic population size",
            default=args.starting,
            min_value=1,
        )

        args.generations = prompt_line(
            "Enter the number of genetic generations per round",
            default=args.generations,
            min_value=1,
        )

        args.elite = prompt_line(
            "Enter the number of elite rules preserved each generation",
            default=args.elite,
            min_value=1,
        )

        args.target_solutions = prompt_target_line(
            "Enter the number of target sound rules to find (or 'inf')",
            default=args.target_solutions,
        )

    # Sanity Check Arguments
    if args.elite > args.starting:
        print(f"error: --elite ({args.elite}) cannot be greater than --starting ({args.starting})", file=sys.stderr)
        return 2

    # Run CEGIS
    cegis = CEGIS(
        form,
        max_rounds=args.max_rounds,
        starting=args.starting,
        generations=args.generations,
        elite=args.elite,
        target_solutions=args.target_solutions,
    )

    rules = cegis.synthesise()

    if len(rules) == 0:
        print("\nNo Pruning Rules found.")

    else:
        print("\nGenerated & Verified Pruning Conditions:")
        for r in rules:
            print(f"\t‣\t{r}")
        return 0





# Function to collect arguments and run CEGIS.
def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)





# If run, assume the default arguments.
if __name__ == "__main__":
    raise SystemExit(main())
