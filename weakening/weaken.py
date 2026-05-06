import datetime

from .ast import *
from .genetic import ProgramSearch
from .genetic import configure_verifier
from .verifier import CompleteOrderVerifier
import sys
from tqdm import trange
from sygus.synth import log



verifier = CompleteOrderVerifier()
set_symbol_universe(verifier.symbol_set())
configure_verifier(verifier)
_MC_CACHE = {}



class Weakener:
    def __init__(self, verifier: CompleteOrderVerifier | None = None, *, max_rounds=50, starting=30,
                 generations=50, elite=4, target_solutions=5, seed_rules=None):
        self.verifier = verifier or CompleteOrderVerifier()
        set_symbol_universe(self.verifier.symbol_set())
        configure_verifier(self.verifier)
        self.max_rounds = max_rounds
        self.starting = starting
        self.generations = generations
        self.elite = elite
        self.target_solutions = target_solutions

        self.verified_rules = set()
        self.seed_rules = list(seed_rules or [])
        self.pop = list(seed_rules or [])
        self.round_number = 0

    def verify_seed_rules(self):
        for seed in self.seed_rules:
            if seed in self.verified_rules:
                continue
            if not self.verifier.is_rule_satisfiable(seed):
                continue
            verification = self.verifier.verify_rule(seed)
            if verification.is_verified:
                self.verified_rules.add(seed)
                continue
            self.seed_rules = [rule for rule in self.seed_rules if str(rule) != str(seed)]
            self.pop = [rule for rule in self.pop if str(rule) != str(seed)]

    def round(self, progress=None):
        self.round_number += 1

        pop, rejected = ProgramSearch.search(
            start=self.starting,
            gens=self.generations,
            pop=self.pop,
            seed_rules=self.seed_rules,
            verifier=self.verifier,
            progress=progress,
        )

        self.pop = pop
        for seed in self.seed_rules:
            if seed not in self.pop:
                self.pop.append(seed)

        new_rules = 0
        for rule in self.pop:
            before = len(self.verified_rules)
            self.verified_rules.add(rule)
            new_rules += len(self.verified_rules) - before

        return new_rules, rejected

    def synthesise(self) -> [BooleanExpr]:
        self.verify_seed_rules()
        log("Invoking Pruning Rule Weakening & Exploration.")
        total_new_rules = 0
        total_rejected = 0
        total_steps = self.max_rounds * self.generations

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        with trange(
            total_steps,
            desc=f"[{timestamp}] Weakening Dominance Condition",
            leave=True,
            colour="green",
            file=sys.stdout,
            bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
        ) as progress:
            for _ in range(self.max_rounds):
                if len(self.verified_rules) >= self.target_solutions:
                    break
                new_rules, rejected = self.round(progress=progress)
                total_new_rules += new_rules
                total_rejected += rejected
                if len(self.verified_rules) >= self.target_solutions:
                    break
        log(
            "Synthesis Complete:"
            f"{total_new_rules} Mutated Rules, "
            f"{len(self.verified_rules)} Total Rules, "
            f"{total_rejected} Rejected Rules."
        )
        return self.verified_rules
