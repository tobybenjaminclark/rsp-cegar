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
                 generations=50, elite=4, target_solutions=5):
        self.verifier = verifier or CompleteOrderVerifier()
        set_symbol_universe(self.verifier.symbol_set())
        configure_verifier(self.verifier)
        self.max_rounds = max_rounds
        self.starting = starting
        self.generations = generations
        self.elite = elite
        self.target_solutions = target_solutions

        self.Σ = []
        self.verified_rules = set()
        self.pop = None
        self.round_number = 0

    def round(self):
        self.round_number += 1

        tqdm.write(f"\n𝗖𝗼𝘂𝗻𝘁𝗲𝗿 𝗘𝘅𝗮𝗺𝗽𝗹𝗲 𝗚𝘂𝗶𝗱𝗲𝗱 𝗜𝗻𝗱𝘂𝗰𝘁𝗶𝘃𝗲 𝗦𝘆𝗻𝘁𝗵𝗲𝘀𝗶𝘀 | Round {self.round_number} of {self.max_rounds} | Solutions Found: {len(self.verified_rules)} of {self.target_solutions} | Σ* contains {len(self.Σ)} counterexamples")

        best, best_score, pop = ProgramSearch.search(
            start=self.starting,
            gens=self.generations,
            elite=self.elite,
            Σ=self.Σ,
            pop=self.pop,
            antiduplication=False
        )

        self.pop = pop

        fitpop = ProgramSearch.fitness(pop, self.Σ)

        top3 = sorted(fitpop, key=lambda x: x[1], reverse=True)[:3]
        for i, (rule, total, sigma, size_pen, entropy_mc) in enumerate(top3, 1):
            tqdm.write(
                f" ► [{i}] {str(rule):<40} :: "
                f" {total:7.4f} | "
                f"Σ:{sigma:7.4f} + "
                f"|β|:{size_pen:7.4f} + "
                f"MC:{entropy_mc:7.4f}"
            )

        if not self.verifier.is_rule_satisfiable(best):
            tqdm.write(" ► Top rule is 𝗩𝗔𝗖𝗨𝗢𝗨𝗦𝗟𝗬-𝗨𝗡𝗦𝗔𝗧𝗜𝗦𝗙𝗜𝗔𝗕𝗟𝗘 (removing from population)")
            self.pop = [r for r in self.pop if str(r) != str(best)]
            return

        verification = self.verifier.verify_rule(best)
        if verification.is_verified:
            tqdm.write(" ► Top rule is 𝗦𝗢𝗨𝗡𝗗 (appending rule into verified-solutions)")
            self.verified_rules.add(best)
            return
        if not verification.is_non_vacuous:
            tqdm.write(" ► Top rule is 𝗩𝗔𝗖𝗨𝗢𝗨𝗦 (removing from population)")
            self.pop = [r for r in self.pop if str(r) != str(best)]
            return
        if verification.counterexample is None:
            tqdm.write(" ► Top rule could not be classified by cvc5 (removing from population)")
            self.pop = [r for r in self.pop if str(r) != str(best)]
            return
        else:
            tqdm.write(" ► Top rule is 𝗨𝗡𝗦𝗢𝗨𝗡𝗗 (appending counter-example into Σ*)")

        self.Σ.append((verification.counterexample, False))

    def synthesise(self) -> [BooleanExpr]:
        for outer in range(self.max_rounds):
            self.round()
            if len(self.verified_rules) >= self.target_solutions:
                break
        return self.verified_rules
