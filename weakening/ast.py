from __future__ import annotations
from enum import Enum
from pydantic import BaseModel
from pydantic.config import ConfigDict
import random
import operator
import cvc5
from cvc5 import Kind
import copy





# Define a trait necessitating an object is evaluable
class Evaluable:
    def eval(self):    raise NotImplementedError()

# Define a trait necessitating an object is evaluable within Numpy
class NumpyEvaluable:
    def eval(self):    raise NotImplementedError()

# Define a trait necessitating an object must convert to SMT within the cvc5 framework
class SMTConvertible:
    def to_cvc5(self, solver: cvc5.Solver, symbols=None):    raise NotImplementedError()

# Define a trait necessitating an object must mutate (and support random creation)
class Genetic:
    def mutate(self):   raise NotImplementedError()

    @classmethod
    def random(cls):    raise NotImplementedError()





# Enumerations to denote comparison and arithmetic operators
class CmpOp(SMTConvertible, str, Enum):
    GT = ">"
    LT = "<"
    EQ = "="
    GE = "≥"
    LE = "≤"

    def to_cvc5(self, solver: cvc5.Solver, left, right):
        match self:
            case CmpOp.GT: return solver.mkTerm(Kind.GT, left, right)
            case CmpOp.LT: return solver.mkTerm(Kind.LT, left, right)
            case CmpOp.EQ: return solver.mkTerm(Kind.EQUAL, left, right)
            case CmpOp.GE: return solver.mkTerm(Kind.GEQ, left, right)
            case CmpOp.LE: return solver.mkTerm(Kind.LEQ, left, right)

    def get_op(self):
        match self:
            case CmpOp.GT: return operator.gt
            case CmpOp.LT: return operator.lt
            case CmpOp.EQ: return operator.eq
            case CmpOp.GE: return operator.ge
            case CmpOp.LE: return operator.le

class ArithOp(SMTConvertible, str, Enum):
    ADD = "+"
    SUB = "-"
    MUL = "×"
    DIV = "÷"

    def to_cvc5(self, solver: cvc5.Solver, left, right):
        match self:
            case ArithOp.ADD: return solver.mkTerm(Kind.ADD, left, right)
            case ArithOp.SUB: return solver.mkTerm(Kind.SUB, left, right)
            case ArithOp.MUL: return solver.mkTerm(Kind.MULT, left, right)
            case ArithOp.DIV: return solver.mkTerm(Kind.DIVISION, left, right)

    def get_op(self):
        match self:
            case ArithOp.ADD: return operator.add
            case ArithOp.SUB: return operator.sub
            case ArithOp.MUL: return operator.mul
            case ArithOp.DIV: return operator.truediv





# Base Expressions
class BooleanExpr(BaseModel, Genetic, SMTConvertible, Evaluable, NumpyEvaluable):
    model_config = ConfigDict(frozen=False)
    def mutate(self):   raise NotImplementedError()

    @classmethod
    def random(_, depth=2): return Cmp.random(depth) if depth <= 0 else random.choice([And, Or, Not, Cmp]).random(depth - 1)

    def __len__(self):          return 1 + sum(len(child) for child in self)
    def eval_np(self, _):       raise NotImplementedError
    def __hash__(self):         return hash(str(self))

    def walk(self):
        yield self
        for child in self:
            yield from child.walk()

class ArithExpr(BaseModel, Genetic, SMTConvertible, Evaluable, NumpyEvaluable):
    model_config = ConfigDict(frozen=False)
    def mutate(self):       raise NotImplementedError()

    @classmethod
    def random(_, depth=1):
        return Symbol.random() if depth <= 0 else random.choice([Binary, Symbol]).random(depth-1)

    def __len__(self):      return 1 + sum(len(child) for child in self)
    def eval_np(self, _):   raise NotImplementedError
    def __hash__(self):     return hash(str(self))

    def walk(self):
        yield self
        for child in self:
            yield from child.walk()





# Boolean Expressions
class And(BooleanExpr):
    left: BooleanExpr
    right: BooleanExpr
    def __init__(self, l: BooleanExpr, r: BooleanExpr):
        super().__init__(left=l, right=r)

    def __str__(self):      return f"{self.left} ∧ {self.right}"
    def __iter__(self):     return iter((self.left, self.right))
    def mutate(self):       return Or(self.left, self.right)
    def eval(self, sample): return self.left.eval(sample) and self.right.eval(sample)
    def eval_np(self, arr): return [left and right for left, right in zip(self.left.eval_np(arr), self.right.eval_np(arr))]

    def to_cvc5(self, solver: cvc5.Solver, symbols=None):
        return solver.mkTerm(Kind.AND, self.left.to_cvc5(solver, symbols), self.right.to_cvc5(solver, symbols))

    @classmethod
    def random(_, depth=2):     return And(BooleanExpr.random(depth-1), BooleanExpr.random(depth-1))


class Or(BooleanExpr):
    left: BooleanExpr
    right: BooleanExpr
    def __init__(self, l: BooleanExpr, r: BooleanExpr):  super().__init__(left=l, right=r)

    def __str__(self):      return f"{self.left} ∨ {self.right}"
    def __iter__(self):     return iter((self.left, self.right))
    def mutate(self):       return And(self.left, self.right)
    def eval(self, sample): return self.left.eval(sample) or self.right.eval(sample)
    def eval_np(self, arr): return [left or right for left, right in zip(self.left.eval_np(arr), self.right.eval_np(arr))]

    def to_cvc5(self, solver: cvc5.Solver, symbols=None):
        return solver.mkTerm(Kind.OR, self.left.to_cvc5(solver, symbols), self.right.to_cvc5(solver, symbols))

    @classmethod
    def random(_, depth=2):
        return Or(BooleanExpr.random(depth-1), BooleanExpr.random(depth-1))


class Not(BooleanExpr):
    inner: BooleanExpr

    def __init__(self, i: BooleanExpr):  super().__init__(inner=i)
    def __str__(self):      return f"¬ ({self.inner})"
    def __iter__(self):     return iter((self.inner,))
    def mutate(self):       return (self.inner)
    def eval(self, sample): return not (self.inner.eval(sample))
    def eval_np(self, arr): return [not value for value in self.inner.eval_np(arr)]

    def to_cvc5(self, solver: cvc5.Solver, symbols=None):
        return solver.mkTerm(Kind.NOT, self.inner.to_cvc5(solver, symbols))

    @classmethod
    def random(_, depth=2):
        return Not(BooleanExpr.random(depth-1))


class Cmp(BooleanExpr):
    left: ArithExpr
    op: CmpOp
    right: ArithExpr

    def __init__(self, l: ArithExpr, o: CmpOp, r: ArithExpr):  super().__init__(left=l, op=o, right=r)

    def __str__(self):      return f"{self.left} {self.op.value} {self.right}"
    def __iter__(self):     return iter((self.left, self.right))
    def mutate(self):       return Cmp(self.left, random.choice(list(CmpOp)), self.right)

    def to_cvc5(self, solver: cvc5.Solver, symbols=None):
        return self.op.to_cvc5(solver, self.left.to_cvc5(solver, symbols), self.right.to_cvc5(solver, symbols))

    def eval(self, sample):
        return self.op.get_op()(self.left.eval(sample), self.right.eval(sample))

    def eval_np(self, arr):
        op = self.op.get_op()
        return [op(left, right) for left, right in zip(self.left.eval_np(arr), self.right.eval_np(arr))]

    @classmethod
    def random(_, depth=0):
        return Cmp(ArithExpr.random(1), random.choice(list(CmpOp)), ArithExpr.random(1))





# Arithmetic Expressions
class Binary(ArithExpr):
    left: ArithExpr
    op: ArithOp
    right: ArithExpr
    def __str__(self):      return f"{self.left} {self.op.value} {self.right}"
    def __init__(self,  l: ArithExpr, o: ArithOp, r: ArithExpr):  super().__init__(left=l, op=o, right=r)
    def __iter__(self):     return iter((self.left, self.right))
    def mutate(self):       return Binary(self.left, random.choice(list(ArithOp)), self.right)

    def to_cvc5(self, solver: cvc5.Solver, symbols=None):
        return self.op.to_cvc5(solver, self.left.to_cvc5(solver, symbols), self.right.to_cvc5(solver, symbols))

    def eval(self, sample):
        left = self.left.eval(sample)
        right = self.right.eval(sample)
        if self.op == ArithOp.DIV and right == 0:
            return float("inf")
        return self.op.get_op()(left, right)

    def eval_np(self, arr):
        op = self.op.get_op()
        left_values = self.left.eval_np(arr)
        right_values = self.right.eval_np(arr)
        if self.op == ArithOp.DIV:
            return [op(left, float("inf") if right == 0 else right) for left, right in zip(left_values, right_values)]
        return [op(left, right) for left, right in zip(left_values, right_values)]


    @classmethod
    def random(_, depth=1):
        return Binary(ArithExpr.random(depth-1), random.choice(list(ArithOp)), ArithExpr.random(depth-1))

class Number(ArithExpr):
    value: float
    def __init__(self,  n: float):  super().__init__(value=n)
    def __str__(self):      return str(self.value)
    def __iter__(self):     return iter(())
    def mutate(self):       return Number(self.value + random.randrange(-50, 50) / 100)
    def eval(self, sample): return float(self.value)
    def eval_np(self, arr): return [float(self.value)] * len(next(iter(arr.values())))

    def to_cvc5(self, solver: cvc5.Solver, symbols=None):
        if float(self.value).is_integer():
            return solver.mkInteger(int(self.value))
        return solver.mkReal(str(self.value))

    @classmethod
    def random(_, depth=0):     return random.choice(SYMBOLS)

class Symbol(ArithExpr):
    iden: str
    def __init__(self,  i: str):  super().__init__(iden=i)
    def __str__(self):      return self.iden
    def __iter__(self):     return iter(())
    def mutate(self):       return random.choice(SYMBOLS)
    def eval(self, sample): return sample[self.iden]
    def eval_np(self, arr): return arr[self.iden]

    def to_cvc5(self, solver: cvc5.Solver, symbols=None):
        if symbols and self.iden in symbols:
            return symbols[self.iden]
        return solver.mkConst(solver.getRealSort(), self.iden)

    @classmethod
    def random(_, depth=0):     return random.choice(SYMBOLS)





def replace_subtree(root, target, repl):
    """Return a tree where one subtree is replaced. Mutates only if needed."""
    if root is target:
        return repl

    for attr, val in root.__dict__.items():
        if val is target:
            setattr(root, attr, copy.deepcopy(repl))
            return root

        if isinstance(val, (BooleanExpr, ArithExpr)):
            new = replace_subtree(val, target, repl)
            if new is not val:
                setattr(root, attr, new)
                return root

    return root





def set_symbol_universe(names):
    global SYMBOLS
    SYMBOLS = [Symbol(n) for n in names]
