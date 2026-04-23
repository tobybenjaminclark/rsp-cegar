from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from typing import Protocol

import cvc5
from cvc5 import Kind

from core.context import RSPContext


class SMTConvertible(Protocol):
    def to_cvc5(self, solver: cvc5.Solver) -> object:
        raise Exception("Not implemented")


SMT_ENV: dict[str, dict[str, object]] = {
    "symbol_table": {},
    "nonterminal_table": {},
}


def set_smt_env(
    *,
    symbol_table: dict[str, object] | None = None,
    nonterminal_table: dict[str, object] | None = None,
) -> None:
    if symbol_table is not None:
        SMT_ENV["symbol_table"] = dict(symbol_table)
    if nonterminal_table is not None:
        SMT_ENV["nonterminal_table"] = dict(nonterminal_table)


class Expr(SMTConvertible):
    def __or__(self, other: Expr) -> "Choice":
        if isinstance(self, Choice): return self.__or__(other)
        return Choice((self, other))

    def __and__(self, other: Expr) -> "And":
        if isinstance(self, And): return And(self.terms + (other,))
        return And((self, other))

    def __add__(self, other: Expr) -> "Add":
        if isinstance(self, Add): return Add(self.terms + (other,))
        return Add((self, other))

    def __le__(self, other: Expr) -> "Leq":
        return Leq(self, other)

    def eq(self, other: Expr) -> "Eq":
        return Eq(self, other)

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        raise NotImplementedError


@dataclass(frozen=True)
class Terminal(Expr):
    name: str
    formal: object | None = None
    actual: object | None = None

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        if self.name in SMT_ENV["symbol_table"]: return SMT_ENV["symbol_table"][self.name]
        if self.formal is not None:         return self.formal
        raise KeyError(f"Unknown terminal symbol: {self.name}")


@dataclass
class NonTerminal(Expr):
    name: str
    grammar: "Grammar | None" = None

    def __rshift__(self, rhs: Expr) -> "Production":
        return Production(self, rhs if isinstance(rhs, Choice) else Choice((rhs,)))

    def __irshift__(self, rhs: Expr) -> "NonTerminal":
        if self.grammar is None:
            raise ValueError(f"NonTerminal {self.name} is not attached to a Grammar.")
        self.grammar.add_production(self >> rhs)
        return self

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        if self.name not in SMT_ENV["nonterminal_table"]:
            raise KeyError(f"Unknown nonterminal symbol: {self.name}")
        return SMT_ENV["nonterminal_table"][self.name]


@dataclass(frozen=True)
class Choice(Expr):
    alts: tuple[Expr, ...]

    def __or__(self, other: Expr) -> "Choice":
        if isinstance(other, Choice): return Choice(self.alts + other.alts)
        return Choice(self.alts + (other,))

    def to_cvc5(self, solver: cvc5.Solver) -> tuple[object, ...]:
        return tuple(map(lambda alt: alt.to_cvc5(solver), self.alts))


@dataclass(frozen=True)
class And(Expr):
    terms: tuple[Expr, ...]

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        compiled = tuple(map(lambda term: term.to_cvc5(solver), self.terms))
        if not compiled:        return solver.mkTrue()
        if len(compiled) == 1:  return compiled[0]
        return solver.mkTerm(Kind.AND, *compiled)


@dataclass(frozen=True)
class Eq(Expr):
    left: Expr
    right: Expr

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        return solver.mkTerm(Kind.EQUAL, self.left.to_cvc5(solver), self.right.to_cvc5(solver))


@dataclass(frozen=True)
class Leq(Expr):
    left: Expr
    right: Expr

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        return solver.mkTerm(Kind.LEQ, self.left.to_cvc5(solver), self.right.to_cvc5(solver))


@dataclass(frozen=True)
class Add(Expr):
    terms: tuple[Expr, ...]

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        compiled = tuple(map(lambda term: term.to_cvc5(solver), self.terms))
        if not compiled:        return solver.mkInteger(0)
        if len(compiled) == 1:  return compiled[0]
        return solver.mkTerm(Kind.ADD, *compiled)


@dataclass(frozen=True)
class Production(SMTConvertible):
    lhs: NonTerminal
    rhs: Choice

    def to_cvc5(self, solver: cvc5.Solver) -> tuple[object, tuple[object, ...]]:
        return self.lhs.to_cvc5(solver), self.rhs.to_cvc5(solver)


@dataclass
class Grammar(SMTConvertible):
    nonterminals: tuple[NonTerminal, ...]
    terminals: tuple[Terminal, ...]
    start: NonTerminal
    productions: tuple[Production, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        attached = tuple(
            map(
                lambda nt: nt if nt.grammar is self else NonTerminal(nt.name, grammar=self),
                self.nonterminals,
            )
        )
        index = {nt.name: nt for nt in attached}
        self.nonterminals = attached
        self.start = index.get(self.start.name, self.start)
        self.productions = tuple(
            map(
                lambda p: Production(index.get(p.lhs.name, p.lhs), p.rhs),
                self.productions,
            )
        )

    def add_production(self, production: Production) -> "Grammar":
        self.productions = self.productions + (production,)
        return self

    def map_productions(self, fn):
        return Grammar(
            nonterminals=self.nonterminals,
            terminals=self.terminals,
            start=self.start,
            productions=tuple(map(fn, self.productions)),
        )

    def reduce_productions(self, fn, init):
        return reduce(fn, self.productions, init)

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        nonterminal_table = dict(SMT_ENV["nonterminal_table"])
        nonterminal_table.update(
            {
                nt.name: nonterminal_table.get(nt.name, solver.mkVar(solver.getBooleanSort(), nt.name))
                for nt in self.nonterminals
            }
        )
        SMT_ENV["nonterminal_table"] = nonterminal_table
        grammar = solver.mkGrammar(
            tuple(map(lambda t: t.to_cvc5(solver), self.terminals)),
            tuple(map(lambda nt: nt.to_cvc5(solver), self.nonterminals)),
        )
        list(
            map(
                lambda p: grammar.addRules(
                    p.to_cvc5(solver)[0],
                    p.to_cvc5(solver)[1],
                ),
                self.productions,
            )
        )
        return grammar


def make_pruning_rule_grammar(env, symbols: tuple[Terminal, ...]):
    cond = NonTerminal("Rule")
    x = NonTerminal("Atom")

    by_name = {symbol.name: symbol for symbol in symbols}
    names = set(by_name)

    prefixes = sorted({name[:-2] for name in names if name.endswith("_i") and not name.startswith("D_")})
    comparable_pairs = [
        *[(f"{p}_i", f"{p}_j") for p in prefixes if f"{p}_j" in names],
        *[pair for pair in (("D_i_x", "D_j_x"), ("D_x_i", "D_x_j")) if pair[0] in names and pair[1] in names],
    ]
    atom_alts = [[by_name[l] <= by_name[r], by_name[r] <= by_name[l], by_name[l].eq(by_name[r])] for l, r in comparable_pairs ]
    flat = lambda xs: [y for x in xs for y in (flat(x) if isinstance(x, list) else [x])]

    grammar = Grammar(
        nonterminals=(cond, x),
        terminals=symbols,
        start=cond,
        productions=(
            cond >> (x | (x & x) | (x & x & x)),
            x >> Choice(tuple(flat(atom_alts))),
        ),
    )
    symbol_table = dict(map(lambda symbol: (symbol.name, symbol.formal), symbols))
    nonterminal_table = {
        "Rule": env.solver.mkVar(env.bool_sort, "Rule"),
        "Atom": env.solver.mkVar(env.bool_sort, "Atom"),
    }
    set_smt_env(symbol_table=symbol_table, nonterminal_table=nonterminal_table)
    return grammar.to_cvc5(env.solver)
