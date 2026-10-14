"""Safe group filters such as ``country == 'USA' and age > 30 and info_tier >= 2``.

Only names from a whitelist, numeric/string constants, comparisons, boolean
operators, arithmetic and ``in`` with tuples/lists are allowed.  Evaluation is
vectorised over all agents.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
from numpy.typing import NDArray

from jahan.model.enums import EmpStatus, Marital, Sex
from jahan.model.world import World
from jahan.sim.decisions import income_basis


class FilterError(ValueError):
    pass


def _columns(world: World) -> dict[str, Any]:
    pop = world.pop
    iso = np.array([c.iso3 for c in world.countries], dtype=object)
    sex_names = np.array([s.name for s in Sex], dtype=object)
    emp_names = np.array([e.name.lower() for e in EmpStatus], dtype=object)
    marital_names = np.array([m.name.lower() for m in Marital], dtype=object)
    return {
        "id": pop["id"],
        "alive": pop["alive"],
        "age": world.ages(),
        "country": iso[pop["country"].astype(np.int64)],
        "location": iso[pop["location_country"].astype(np.int64)],
        "sex": sex_names[pop["sex"].astype(np.int64)],
        "urban": pop["region"] == 1,
        "education": pop["education"].astype(np.int64),
        "occupation": pop["occupation"].astype(np.int64),
        "status": emp_names[pop["emp_status"].astype(np.int64)],
        "marital": marital_names[pop["marital"].astype(np.int64)],
        "income": income_basis(world),
        "info_tier": pop["info_tier"].astype(np.int64),
        "internet": pop["internet"],
        "visibility": pop["visibility"].astype(np.float64),
        "honesty": pop["honesty"].astype(np.float64),
        "risk": pop["risk_tolerance"].astype(np.float64),
        "notable": pop["is_notable"],
        "household": pop["household_id"].astype(np.int64),
        "tz_offset_min": pop["tz_offset_min"].astype(np.int64),
    }


_CMP = {
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}


def evaluate(world: World, expr: str) -> NDArray[np.bool_]:
    cols = _columns(world)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise FilterError(f"bad filter: {e}") from e

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [np.asarray(ev(v), dtype=bool) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ~np.asarray(ev(node.operand), dtype=bool)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            acc: Any = None
            for op, comp in zip(node.ops, node.comparators, strict=True):
                right = ev(comp)
                if isinstance(op, ast.In | ast.NotIn):
                    r = np.isin(left, list(right))
                    r = ~r if isinstance(op, ast.NotIn) else r
                elif type(op) in _CMP:
                    r = _CMP[type(op)](left, right)
                else:
                    raise FilterError(f"operator {type(op).__name__} not allowed")
                acc = r if acc is None else (acc & r)
                left = right
            return acc
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Name):
            if node.id not in cols:
                raise FilterError(f"unknown field '{node.id}'; fields: {sorted(cols)}")
            return cols[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float | str | bool):
            return node.value
        if isinstance(node, ast.Tuple | ast.List):
            return [ev(e) for e in node.elts]
        raise FilterError(f"expression element {type(node).__name__} not allowed")

    result = np.asarray(ev(tree), dtype=bool)
    if result.shape != (world.pop.n,):
        result = np.broadcast_to(result, (world.pop.n,)).copy()
    return result
