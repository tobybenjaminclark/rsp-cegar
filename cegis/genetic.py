from .ast import *
from .verifier import CompleteOrderVerifier
import copy
import functools
import random
import time
from tqdm import trange, tqdm
import sys





def timed(name, fn):
    start = time.perf_counter()
    out = fn()
    TIMINGS[name] = TIMINGS.get(name, 0) + (time.perf_counter() - start)
    return out



@functools.lru_cache(None)
def _equiv_cvc5(a, b):
    return VERIFIER.equivalent(a, b)

TIMINGS = {}
VERIFIER = CompleteOrderVerifier()


def configure_verifier(verifier: CompleteOrderVerifier) -> None:
    global VERIFIER
    VERIFIER = verifier
    _equiv_cvc5.cache_clear()





class ProgramSearch:

    @staticmethod
    def unique(pop: list[BooleanExpr]) -> list[BooleanExpr]:
        kept = []
        seen = set()
        for rule in pop:
            key = str(rule)
            if key not in seen:
                kept.append(rule)
                seen.add(key)
        return kept

    @staticmethod
    def _collect(expr, kinds):
        """ Collect nodes of a pruning condition """
        return [n for n in expr.walk() if isinstance(n, kinds)]


    @staticmethod
    def find_parent(expr, target):
        """ Retrieve the parent of a pruning condition """
        return next(((p, k) for p in expr.walk() for k, v in p.__dict__.items() if v is target), (None, None))


    @staticmethod
    def breed(a: BooleanExpr, b: BooleanExpr) -> BooleanExpr:
        """Join two predicates in the Boolean lattice.

        The result is a weakening of both parents because a => (a or b).
        """
        return Or(copy.deepcopy(a), copy.deepcopy(b))


    @staticmethod
    def _conjuncts(expr: BooleanExpr) -> list[BooleanExpr]:
        if isinstance(expr, And):
            return ProgramSearch._conjuncts(expr.left) + ProgramSearch._conjuncts(expr.right)
        return [expr]


    @staticmethod
    def _and_all(terms: list[BooleanExpr]) -> BooleanExpr:
        if not terms:
            return ProgramSearch.tautology()
        return functools.reduce(lambda left, right: And(left, right), terms)


    @staticmethod
    def tautology() -> BooleanExpr:
        return Cmp(Symbol("R_i"), CmpOp.LE, Symbol("R_i"))


    @staticmethod
    def random_atom() -> BooleanExpr:
        return Cmp.random(0)


    @staticmethod
    def weaken_cmp(expr: Cmp) -> BooleanExpr:
        match expr.op:
            case CmpOp.EQ:
                return Cmp(copy.deepcopy(expr.left), random.choice([CmpOp.LE, CmpOp.GE]), copy.deepcopy(expr.right))
            case CmpOp.LT:
                return Cmp(copy.deepcopy(expr.left), CmpOp.LE, copy.deepcopy(expr.right))
            case CmpOp.GT:
                return Cmp(copy.deepcopy(expr.left), CmpOp.GE, copy.deepcopy(expr.right))
            case CmpOp.LE | CmpOp.GE:
                return ProgramSearch.tautology()


    @staticmethod
    def weaken(expr: BooleanExpr) -> BooleanExpr:
        """Return a predicate implied by expr, i.e. a lattice weakening."""
        expr = copy.deepcopy(expr)

        if isinstance(expr, And):
            terms = ProgramSearch._conjuncts(expr)
            if len(terms) > 1 and random.random() < 0.5:
                del terms[random.randrange(len(terms))]
                return ProgramSearch._and_all(terms)

            index = random.randrange(len(terms))
            terms[index] = ProgramSearch.weaken(terms[index])
            return ProgramSearch._and_all(terms)

        if isinstance(expr, Or):
            choices = [
                lambda: Or(expr, ProgramSearch.random_atom()),
                lambda: Or(ProgramSearch.weaken(expr.left), expr.right),
                lambda: Or(expr.left, ProgramSearch.weaken(expr.right)),
            ]
            return random.choice(choices)()

        if isinstance(expr, Not):
            return Or(expr, ProgramSearch.random_atom())

        if isinstance(expr, Cmp):
            return ProgramSearch.weaken_cmp(expr)

        return Or(expr, ProgramSearch.random_atom())


    @staticmethod
    def mutate_one(expr):
        """Mutate by weakening only, preserving parent => child."""
        return ProgramSearch.weaken(expr)


    @staticmethod
    def gen_initial(n: int):
        """ Generate an initial population of `n` boolean expressions."""
        return [BooleanExpr.random(random.choice([1, 2, 3, 4])) for _ in range(n)]

    @staticmethod
    def gen_seeded(seed_rules: list[BooleanExpr], n: int):
        """Generate an initial population by weakening supplied seed expressions."""
        if not seed_rules:
            return ProgramSearch.gen_initial(n)

        pop = [copy.deepcopy(seed) for seed in seed_rules[:n]]
        while len(pop) < n:
            seed = copy.deepcopy(random.choice(seed_rules))
            pop.append(ProgramSearch.mutate_one(seed))
        return pop


    @staticmethod
    def selection(pop, start, seed_rules=None):
        """Select sound parents without any numeric objective."""
        survivors = ProgramSearch.unique(list(pop))
        random.shuffle(survivors)
        for seed in seed_rules or []:
            if not any(str(seed) == str(survivor) for survivor in survivors):
                survivors.append(copy.deepcopy(seed))
        survivors = survivors[:start]
        return survivors + [None] * (start - len(survivors))


    @staticmethod
    def crossover(pop):
        """Replace empty slots with lattice joins of selected parents."""
        if len(list(filter(lambda p: p is not None, pop))) < 2: return pop
        else: return [v or ProgramSearch.breed(*random.sample([p for p in pop if p], 2)) for v in pop]


    @staticmethod
    def mutation(pop, chance=0.5):
        """Mutate a population using weakening-only moves."""
        mutpop = [ProgramSearch.mutate_one(p) if random.random() < chance and p is not None else p for p in pop]
        parents = [p for p in pop if p is not None]
        return [
            value if value is not None else ProgramSearch.mutate_one(random.choice(parents))
            for value in mutpop
        ]

    @staticmethod
    def is_sound(rule: BooleanExpr, verifier: CompleteOrderVerifier) -> bool:
        return verifier.is_rule_satisfiable(rule) and verifier.verify_rule(rule).is_verified


    @staticmethod
    def filter_sound(pop: list[BooleanExpr], verifier: CompleteOrderVerifier) -> tuple[list[BooleanExpr], int]:
        sound = []
        rejected = 0
        for rule in ProgramSearch.unique(pop):
            if ProgramSearch.is_sound(rule, verifier):
                sound.append(rule)
            else:
                rejected += 1
        return sound, rejected


    @staticmethod
    def replenish(pop: list[BooleanExpr], start: int, seed_rules: list[BooleanExpr], verifier: CompleteOrderVerifier):
        sources = pop or seed_rules
        attempts = 0
        while sources and len(pop) < start and attempts < start * 20:
            attempts += 1
            candidate = ProgramSearch.mutate_one(random.choice(sources))
            if not any(str(candidate) == str(rule) for rule in pop) and ProgramSearch.is_sound(candidate, verifier):
                pop.append(candidate)
                sources = pop
        return pop


    @staticmethod
    def run_generation(pop, verifier: CompleteOrderVerifier, start=10, seed_rules=None):
        """Generate lattice weakenings and keep only verified-sound rules."""
        seed_rules = seed_rules or []
        surpop = ProgramSearch.selection(pop, start, seed_rules=seed_rules)
        sexpop = timed("crossover", lambda: ProgramSearch.crossover(surpop))
        mutpop = timed("mutation", lambda: ProgramSearch.mutation(sexpop))

        for seed in seed_rules or []:
            if not any(str(seed) == str(expr) for expr in mutpop):
                mutpop[-1] = copy.deepcopy(seed)

        sound, rejected = ProgramSearch.filter_sound(mutpop, verifier)
        sound = ProgramSearch.replenish(sound, start, seed_rules, verifier)
        return sound, rejected


    @staticmethod
    def search(start=10, gens=1000, pop=None, seed_rules=None, verifier=None):
        seed_rules = seed_rules or []
        verifier = verifier or VERIFIER

        total_rejected = 0

        if pop is None:         pop = ProgramSearch.gen_seeded(seed_rules, start)
        elif len(pop) < start:  pop = pop + ProgramSearch.gen_seeded(seed_rules, start - len(pop))
        pop, rejected = ProgramSearch.filter_sound(pop, verifier)
        total_rejected += rejected
        pop = ProgramSearch.replenish(pop, start, seed_rules, verifier)

        with trange(gens, desc=" ► 𝗥𝘂𝗻𝗻𝗶𝗻𝗴 𝗣𝗿𝗼𝗴𝗿𝗮𝗺 𝗦𝗲𝗮𝗿𝗰𝗵", leave=True, file=sys.stdout, colour='green', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as bar:
            for g in bar:
                pop, rejected = ProgramSearch.run_generation(pop, verifier, start=start, seed_rules=seed_rules)
                total_rejected += rejected

        tqdm.write(f" ► Rejected {total_rejected} unsound/vacuous weakened candidates")
        return pop
