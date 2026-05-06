from .ast import *
from .genetic import ProgramSearch
from .genetic import configure_verifier
from .verifier import CompleteOrderVerifier
from tqdm import tqdm



verifier = CompleteOrderVerifier()
set_symbol_universe(verifier.symbol_set())
configure_verifier(verifier)
_MC_CACHE = {}



class CEGIS:
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
            tqdm.write(f" ► Seed rule: {seed}")
            if seed in self.verified_rules:
                continue
            if not self.verifier.is_rule_satisfiable(seed):
                tqdm.write(" ► Seed rule is 𝗩𝗔𝗖𝗨𝗢𝗨𝗦𝗟𝗬-𝗨𝗡𝗦𝗔𝗧𝗜𝗦𝗙𝗜𝗔𝗕𝗟𝗘 (keeping as search seed only)")
                continue
            verification = self.verifier.verify_rule(seed)
            if verification.is_verified:
                tqdm.write(" ► Seed rule is 𝗦𝗢𝗨𝗡𝗗 (saving as verified solution)")
                self.verified_rules.add(seed)
                continue
            tqdm.write(" ► Seed rule is not sound (removing from search seeds)")
            self.seed_rules = [rule for rule in self.seed_rules if str(rule) != str(seed)]
            self.pop = [rule for rule in self.pop if str(rule) != str(seed)]

    def round(self):
        self.round_number += 1

        tqdm.write(f"\n𝗩𝗲𝗿𝗶𝗳𝗶𝗲𝗱 𝗟𝗮𝘁𝘁𝗶𝗰𝗲 𝗪𝗲𝗮𝗸𝗲𝗻𝗶𝗻𝗴 | Round {self.round_number} of {self.max_rounds} | Solutions Found: {len(self.verified_rules)} of {self.target_solutions}")

        pop = ProgramSearch.search(
            start=self.starting,
            gens=self.generations,
            pop=self.pop,
            seed_rules=self.seed_rules,
            verifier=self.verifier,
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

        tqdm.write(f" ► Kept {len(self.pop)} verified-sound weakened candidates ({new_rules} new)")
        for i, rule in enumerate(self.pop[:3], 1):
            tqdm.write(f" ► [{i}] {rule}")

    def synthesise(self) -> [BooleanExpr]:
        self.verify_seed_rules()
        for outer in range(self.max_rounds):
            if len(self.verified_rules) >= self.target_solutions:
                break
            self.round()
            if len(self.verified_rules) >= self.target_solutions:
                break
        return self.verified_rules
