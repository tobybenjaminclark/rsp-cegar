from .form import *
from .pr_ast import *
import random, copy, numpy as np, functools, time
from tqdm import trange, tqdm
from itertools import groupby
import sys





def timed(name, fn):
    start = time.perf_counter()
    out = fn()
    TIMINGS[name] = TIMINGS.get(name, 0) + (time.perf_counter() - start)
    return out





@functools.lru_cache(None)
def mc_env(n=100_000, low=0, high=1_000):
    syms = list(FORM.symbol_set())
    return {
        s: np.random.uniform(low, high, n)
        for s in syms
    }

@functools.lru_cache(None)
def mc_signature(rule, n=100_000):
    env = mc_env(n)
    return rule.eval_np(env).astype(np.bool_)

def sig_hash(sig):
    return hash(sig.tobytes())

@functools.lru_cache(None)
def _equiv_z3(a_z3, b_z3):
    s = z3.Solver()
    s.add(z3.Not(a_z3 == b_z3))
    return s.check() == z3.unsat

@functools.lru_cache(None)
def monte_carlo(rule, n=100_000, low=0, high=1_000):
    env = mc_env(n, low, high)
    return np.mean(rule.eval_np(env))

def entropy(p, eps=1e-9):
    return -(p*np.log(p+eps) + (1-p)*np.log(1-p+eps))





TIMINGS = {}
FORM = CompleteOrderForm()





class ProgramSearch:

    @staticmethod
    def antiduplicate(fitpop):
        """ Removes duplicate conditions, equivalence classes grouped by monte-carlo hashing then independently verified """

        fitpop = sorted(fitpop, key=lambda c: sig_hash(mc_signature(c[0])))
        groups = groupby(fitpop, key=lambda c: sig_hash(mc_signature(c[0])))

        def filter_bucket(bucket):
            representatives = []
            for β in sorted(bucket, key=lambda x: x[1], reverse=True):
                if not any(str(rep) == str(β[0]) or _equiv_z3(β[0].to_z3(), rep.to_z3())for rep in representatives):
                    representatives.append(β[0])
                    yield β

        kept = list(chain.from_iterable(filter_bucket(g) for _, g in groups))
        return kept, len(fitpop) - len(kept)

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
        """Cross two trees at compatible AST nodes."""
        a = copy.deepcopy(a)
        nodes_a = ProgramSearch._collect(a, (BooleanExpr, ArithExpr))
        nodes_b = ProgramSearch._collect(copy.deepcopy(b), (BooleanExpr, ArithExpr))

        cut_a = random.choice(nodes_a)

        same_type = [n for n in nodes_b if isinstance(n, type(cut_a))]
        same_kind = [n for n in nodes_b if isinstance(n, BooleanExpr) == isinstance(cut_a, BooleanExpr)]
        cut_b = random.choice(same_type or same_kind or nodes_b)

        return replace_subtree(a, cut_a, cut_b)


    @staticmethod
    def mutate_one(expr):
        """Mutate an expression somehow"""

        node = random.choice(ProgramSearch._collect(expr, (BooleanExpr, ArithExpr)))
        parent, field = ProgramSearch.find_parent(expr, node)

        # 30%: full replacement
        if random.random() < 0.3:
            repl = node.__class__.random()
            return repl if parent is None else (setattr(parent, field, repl) or expr)

        # 70%: local mutate()
        new = node.mutate()
        if not new or new is node:
            return expr
        return new if parent is None else (setattr(parent, field, new) or expr)


    @staticmethod
    def gen_initial(n: int):
        """ Generate an initial population of `n` boolean expressions."""
        return [BooleanExpr.random(random.choice([1, 2, 3, 4])) for _ in range(n)]


    @staticmethod
    def selection(fitpop):
        """ Keep top 25% by fitness; others replaced with None for breeding. """
        ranked = sorted(fitpop, key=lambda x: x[1], reverse=True)
        return [expr for expr, *_ in ranked[:len(ranked) // 4]] + [None] * (len(ranked) - len(ranked) // 4)


    @staticmethod
    def crossover(pop):
        """ Perform crossover on a population of expressions, replaces 'None' members with children """
        if len(list(filter(lambda p: p is not None, pop))) < 2: return pop
        else: return [v or ProgramSearch.breed(*random.sample([p for p in pop if p], 2)) for v in pop]


    @staticmethod
    def mutation(pop, chance=0.5):
        """ Mutate a population of expressions """
        mutpop = [ProgramSearch.mutate_one(p) if random.random() < chance and p is not None else p for p in pop]
        return list(map(lambda x: x if x is not None else BooleanExpr.random(random.choice([1, 2, 3, 4])), mutpop))

    @staticmethod
    def _fitness(β: BooleanExpr, Σ: [z3.Model], βmax: int) -> (float, float, float):
        """ Compute fitness for a singular boolean expression. """
        βz = β.to_z3()
        return (
            (sum(z3.is_true(m.eval(βz, model_completion=True)) == e for m, e in Σ) / len(Σ)) if Σ else 0.5,
            1 - len(β) / βmax,
            entropy(monte_carlo(β))
        )


    @staticmethod
    def fitness(pop: [BooleanExpr], Σ) -> [float]:
        """ Compute weighted fitness for a generation of boolean expressions. """
        βmax = len(max(pop, key=len))
        ω = (4.0, 1.0, 1.0)
        return [
            (β, (ω[0] * t1 + ω[1] * t2 + ω[2] * t3) / sum(ω), t1, t2, t3)
            for β in pop
            for (t1, t2, t3) in (ProgramSearch._fitness(β, Σ, βmax),)
        ]


    @staticmethod
    def run_generation(pop, Σ, antiduplication, elite=2):
        """Run one generation step and return the next population."""

        fitpop = ProgramSearch.fitness(pop, Σ)

        if antiduplication: fitpop, _rem = ProgramSearch.antiduplicate(fitpop)
        else: _rem                       = -1

        elites = [e for e, *_ in sorted(fitpop, key=lambda x: x[1], reverse=True)[:elite]]

        surpop = ProgramSearch.selection(fitpop)
        sexpop = timed("crossover", lambda: ProgramSearch.crossover(surpop))
        mutpop = timed("mutation", lambda: ProgramSearch.mutation(sexpop))

        mutpop[:elite] = elites
        return mutpop, fitpop, _rem


    @staticmethod
    def search(start=10, gens=1000, elite=2, Σ=None, pop=None, antiduplication=True):
        Σ = Σ or []

        duplicates_removed = 0

        # See if CEGIS has removed any population
        if pop is None:         pop = ProgramSearch.gen_initial(start)
        elif len(pop) < start:  pop = pop + ProgramSearch.gen_initial(start - len(pop))

        with trange(gens, desc=" ► 𝗥𝘂𝗻𝗻𝗶𝗻𝗴 𝗣𝗿𝗼𝗴𝗿𝗮𝗺 𝗦𝗲𝗮𝗿𝗰𝗵", leave=True, file=sys.stdout, colour='green', bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') as bar:
            for g in bar:
                pop, fitpop, _rem = ProgramSearch.run_generation(pop, Σ, antiduplication, elite=elite)
                duplicates_removed += _rem

        if antiduplication:
            tqdm.write(f" ► Removed {duplicates_removed} duplicate expressions over {gens} rounds, ({duplicates_removed/gens} per generation or {duplicates_removed/(start*gens)}%)")

        best = max(ProgramSearch.fitness(pop, Σ), key=lambda x: x[1])
        return best[0], best[1], pop
