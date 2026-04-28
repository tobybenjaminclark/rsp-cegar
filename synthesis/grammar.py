from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from functools import reduce
import re
from typing import Iterator
from typing import Protocol
import cvc5
from cvc5 import Kind



@dataclass(frozen=True)
class VisContext:
    use_color: bool
    nonterminal_order: tuple[str, ...]
    palette: tuple[str, ...] = ("\033[96m", "\033[92m", "\033[93m", "\033[95m", "\033[94m", "\033[91m")
    reset: str = "\033[0m"
    gray: str = "\033[90m"

    def color_nonterminal(self, name: str) -> str:
        label = f"<{name}>"
        if not self.use_color: return label
        color_index = {nt_name: i for i, nt_name in enumerate(self.nonterminal_order)}
        index = color_index.get(name, len(color_index))
        return f"{self.palette[index % len(self.palette)]}{label}{self.reset}"

    def gray_text(self, text: str) -> str:
        if not self.use_color: return text
        return f"{self.gray}{text}{self.reset}"





SMT_ENV: dict[str, dict[str, object]] = {"symbol_table": {}}

def set_smt_env(*, symbol_table: dict[str, object] | None = None) -> None:
    if symbol_table is not None:
        SMT_ENV["symbol_table"] = dict(symbol_table)





class SMTConvertible(Protocol):
    def to_cvc5(self, solver: cvc5.Solver) -> object:               raise Exception("Not implemented")

class Rebindable(Protocol):
    def __iter__(self) -> Iterator["Expr"]:                         raise Exception("Not implemented")
    def rebind(self, index: dict[str, "NonTerminal"]) -> None:      raise Exception("Not implemented")

class Visualisable(Protocol):
    def vis(self, ctx: VisContext | None = None) -> str:             raise Exception("Not implemented")





class Expr(SMTConvertible, Rebindable, Visualisable):
    def __or__(self, other: Expr) -> "Choice":
        if isinstance(self, Choice): return self.__or__(other)
        return Choice((self, other))

    # Operator overloads for ^ + - <= =
    def __and__(self, other: Expr) -> "And":    return And(self.terms + (other,)) if isinstance(self, And) else And((self, other))
    def __add__(self, other: Expr) -> "Add":    return Add(self.terms + (other,)) if isinstance(self, Add) else Add((self, other))
    def __sub__(self, other: Expr) -> "Sub":    return Sub(self, other)
    def __le__(self, other: Expr) -> "Leq":     return Leq(self, other)
    def eq(self, other: Expr) -> "Eq":          return Eq(self, other)

    def __iter__(self) -> Iterator["Expr"]:
        return iter(())

    def rebind(self, index: dict[str, "NonTerminal"]) -> None:
        for child in self:
            child.rebind(index)

    def vis(self, ctx: VisContext | None = None) -> str:
        return str(self)



@dataclass(frozen=True)
class Terminal(Expr):
    name: str
    formal: object | None = None
    actual: object | None = None

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        if self.name in SMT_ENV["symbol_table"]: return SMT_ENV["symbol_table"][self.name]
        if self.formal is not None:         return self.formal
        raise KeyError(f"Unknown terminal symbol: {self.name}")

    def vis(self, ctx: VisContext | None = None) -> str:
        return self.name



@dataclass
class NonTerminal(Expr):
    name: str
    sort: object | None = None
    grammar: "Grammar | None" = None
    _term: object | None = field(default=None, init=False, repr=False, compare=False)
    _bound_to: "NonTerminal | None" = field(default=None, init=False, repr=False, compare=False)

    def __rshift__(self, rhs: Expr) -> "Production":
        return Production(self, rhs if isinstance(rhs, Choice) else Choice((rhs,)))

    def __irshift__(self, rhs: Expr) -> "NonTerminal":
        if self.grammar is None:
            raise ValueError(f"NonTerminal {self.name} is not attached to a Grammar.")
        self.grammar.add_production(self >> rhs)
        return self

    def _root(self) -> "NonTerminal":
        target = self
        while target._bound_to is not None:
            target = target._bound_to
        return target

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        target = self._root()
        if target._term is None:
            target._term = solver.mkVar(target.sort or solver.getBooleanSort(), target.name)
        return target._term

    def rebind(self, index: dict[str, "NonTerminal"]) -> None:
        target = index.get(self.name)
        if target is not None and target is not self:
            self._bound_to = target

    def vis(self, ctx: VisContext | None = None) -> str:
        if ctx is None:
            return f"<{self.name}>"
        return ctx.color_nonterminal(self.name)



@dataclass(frozen=True)
class Choice(Expr):
    alts: tuple[Expr, ...]

    def __or__(self, other: Expr) -> "Choice":
        if isinstance(other, Choice): return Choice(self.alts + other.alts)
        return Choice(self.alts + (other,))

    def __iter__(self) -> Iterator["Expr"]:
        return iter(self.alts)

    def to_cvc5(self, solver: cvc5.Solver) -> tuple[object, ...]:
        return tuple(map(lambda alt: alt.to_cvc5(solver), self.alts))

    def vis(self, ctx: VisContext | None = None) -> str:
        return " | ".join(alt.vis(ctx) for alt in self.alts)



@dataclass(frozen=True)
class And(Expr):
    terms: tuple[Expr, ...]

    def __iter__(self) -> Iterator["Expr"]:
        return iter(self.terms)

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        compiled = tuple(map(lambda term: term.to_cvc5(solver), self.terms))
        if not compiled:        return solver.mkTrue()
        if len(compiled) == 1:  return compiled[0]
        return solver.mkTerm(Kind.AND, *compiled)

    def vis(self, ctx: VisContext | None = None) -> str:
        return " & ".join(term.vis(ctx) for term in self.terms)



@dataclass(frozen=True)
class Eq(Expr):
    left: Expr
    right: Expr

    def __iter__(self) -> Iterator["Expr"]:
        return iter((self.left, self.right))

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        return solver.mkTerm(Kind.EQUAL, self.left.to_cvc5(solver), self.right.to_cvc5(solver))

    def vis(self, ctx: VisContext | None = None) -> str:
        return f"{self.left.vis(ctx)} = {self.right.vis(ctx)}"



@dataclass(frozen=True)
class Leq(Expr):
    left: Expr
    right: Expr

    def __iter__(self) -> Iterator["Expr"]:
        return iter((self.left, self.right))

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        return solver.mkTerm(Kind.LEQ, self.left.to_cvc5(solver), self.right.to_cvc5(solver))

    def vis(self, ctx: VisContext | None = None) -> str:
        return f"{self.left.vis(ctx)} <= {self.right.vis(ctx)}"



@dataclass(frozen=True)
class Add(Expr):
    terms: tuple[Expr, ...]

    def __iter__(self) -> Iterator["Expr"]:
        return iter(self.terms)

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        compiled = tuple(map(lambda term: term.to_cvc5(solver), self.terms))
        if not compiled:        return solver.mkInteger(0)
        if len(compiled) == 1:  return compiled[0]
        return solver.mkTerm(Kind.ADD, *compiled)

    def vis(self, ctx: VisContext | None = None) -> str:
        return " + ".join(term.vis(ctx) for term in self.terms)



@dataclass(frozen=True)
class Sub(Expr):
    left: Expr
    right: Expr

    def __iter__(self) -> Iterator["Expr"]:
        return iter((self.left, self.right))

    def to_cvc5(self, solver: cvc5.Solver) -> object:
        return solver.mkTerm(Kind.SUB, self.left.to_cvc5(solver), self.right.to_cvc5(solver))

    def vis(self, ctx: VisContext | None = None) -> str:
        return f"{self.left.vis(ctx)} - {self.right.vis(ctx)}"



@dataclass(frozen=True)
class Production(SMTConvertible):
    lhs: NonTerminal
    rhs: Choice

    def __iter__(self) -> Iterator[Expr]:       return iter((self.lhs, self.rhs))
    def to_cvc5(self, solver: cvc5.Solver):     return self.lhs.to_cvc5(solver), self.rhs.to_cvc5(solver)
    def rebind(self, index) -> None:
        self.lhs.rebind(index)
        self.rhs.rebind(index)

    def vis(self, ctx: VisContext | None = None) -> str:
        return f"{self.lhs.vis(ctx)} ::= {self.rhs.vis(ctx)}"





@dataclass
class Grammar(SMTConvertible):
    nonterminals: tuple[NonTerminal, ...]
    terminals: tuple[Terminal, ...]
    start: NonTerminal
    productions: tuple[Production, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        attached = tuple(
            map(
                lambda nt: nt if nt.grammar is self else NonTerminal(nt.name, sort=nt.sort, grammar=self),
                self.nonterminals,
            )
        )
        index = {nt.name: nt for nt in attached}
        self.nonterminals = attached
        self.start = index.get(self.start.name, self.start)
        for production in self.productions:
            production.rebind(index)

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

    def __str__(self) -> str:
        return self.vis()

    def vis(self, ctx: VisContext | None = None, *, use_color: bool = True, max_width: int = 120) -> str:
        if not self.productions:
            return "Grammar"

        local_ctx = ctx or VisContext(use_color=use_color, nonterminal_order=tuple(nt.name for nt in self.nonterminals))
        ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")

        def visible_len(text: str) -> int:
            return len(ansi_pattern.sub("", text))

        lines: list[str] = []
        sep = local_ctx.gray_text(" | ")
        assign = local_ctx.gray_text(" ::= ")
        for production in self.productions:
            lhs = production.lhs.vis(local_ctx)
            rhs = sep.join(alt.vis(local_ctx) for alt in production.rhs.alts)
            prefix = f"{lhs}{assign}"
            continuation_prefix = " " * len(f"<{production.lhs.name}> ::= ")

            current = prefix
            current_len = visible_len(prefix)
            for token in rhs.split(" "):
                piece = token if current_len == visible_len(prefix) or current.endswith(" ") else f" {token}"
                piece_len = visible_len(piece)

                if current_len + piece_len <= max_width:
                    current += piece
                    current_len += piece_len
                else:
                    lines.append(current)
                    current = continuation_prefix + token
                    current_len = visible_len(current)

            lines.append(current)

        return "\n".join(lines)

    def render_tree(self, *, use_color: bool = True) -> str:
        return self.vis(use_color=use_color)

    def to_cvc5(self, solver: cvc5.Solver) -> object:
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
