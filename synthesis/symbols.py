from core import RSPContext
from synthesis import Terminal

_SYGUS_SCHEMA_FIELDS = (
    ("R_i", "r", "i"),
    ("R_j", "r", "j"),
    ("LT_i", "lt", "i"),
    ("LT_j", "lt", "j"),
    ("LC_i", "lc", "i"),
    ("LC_j", "lc", "j"),
    ("ET_i", "et", "i"),
    ("ET_j", "et", "j"),
    ("EC_i", "ec", "i"),
    ("EC_j", "ec", "j"),
    ("B_i", "b", "i"),
    ("B_j", "b", "j"),
    ("C_i", "c", "i"),
    ("C_j", "c", "j"),
)
_SYGUS_DELTA_FIELDS = ("D_i_x", "D_j_x", "D_x_i", "D_x_j")


def make_allowed_symbols(ctx: RSPContext) -> tuple[Terminal, ...]:
    lifted = tuple(
        map(
            lambda spec: Terminal(name=spec[0], formal=ctx.solver.mkVar(ctx.real_sort, spec[0]), actual=getattr(ctx, spec[1])[spec[2]]),
            _SYGUS_SCHEMA_FIELDS,
        )
    )
    delta = tuple(
        map(
            lambda name: Terminal(name=name, formal=ctx.solver.mkVar(ctx.real_sort, name), actual=None),
            _SYGUS_DELTA_FIELDS,
        )
    )
    return lifted + delta