from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import replace
from fractions import Fraction

from .passes_dataclasses import _Canonical, _MatchingState


def _node(
    state: _MatchingState,
    op: str,
    arguments: tuple[_Canonical, ...] = (),
    value: Hashable | None = None,
) -> _Canonical:
    signature = op, tuple(argument.key for argument in arguments), value
    key = state.normal_keys.get(signature)
    if key is None:
        if state.normal_budget <= 0:
            raise OverflowError("canonical expression budget exceeded")
        state.normal_budget -= 1
        key = len(state.normal_keys)
        state.normal_keys[signature] = key
    lower: float | None = None
    upper: float | None = None
    finite = False
    if op == "constant" and isinstance(value, Fraction):
        try:
            lower = upper = float(value)
            finite = math.isfinite(lower)
        except OverflowError:
            pass
    elif op == "add":
        if all(a.lower is not None for a in arguments):
            lower = sum(a.lower for a in arguments if a.lower is not None)
        if all(a.upper is not None for a in arguments):
            upper = sum(a.upper for a in arguments if a.upper is not None)
        finite = all(a.finite for a in arguments)
    elif op == "mul":
        low = high = 1.0
        known = True
        for a in arguments:
            if a.lower is None or a.upper is None:
                known = False
                break
            products = (low * a.lower, low * a.upper, high * a.lower, high * a.upper)
            low, high = min(products), max(products)
        if known:
            lower, upper = low, high
        finite = all(a.finite for a in arguments)
    elif op == "abs":
        lower = 0.0
        a = arguments[0]
        if a.lower is not None and a.upper is not None:
            upper = max(abs(a.lower), abs(a.upper))
        finite = a.finite
    elif op == "exp":
        lower = 0.0
        finite = arguments[0].finite
    elif op in {"sin", "cos", "erf"}:
        lower, upper = -1.0, 1.0
        finite = arguments[0].finite
    elif op == "power" and isinstance(value, Fraction):
        base = arguments[0]
        if value.denominator == 1 and value.numerator % 2 == 0:
            lower = 0.0
        if base.lower is not None and base.lower >= 0 and value > 0:
            lower = 0.0 if base.lower == 0 else base.lower ** float(value)
        finite = (
            base.finite
            and (value >= 0 or _nonzero(base))
            and (value.denominator == 1 or _nonnegative(base))
        )
    elif op in {"min", "max"}:
        if all(a.lower is not None for a in arguments):
            bounds = [a.lower for a in arguments if a.lower is not None]
            lower = min(bounds) if op == "min" else max(bounds)
        if all(a.upper is not None for a in arguments):
            bounds = [a.upper for a in arguments if a.upper is not None]
            upper = min(bounds) if op == "min" else max(bounds)
        finite = all(a.finite for a in arguments)
    elif op == "compare":
        lower, upper, finite = 0.0, 1.0, True
    return _Canonical(
        op, arguments, value, key, lower=lower, upper=upper, finite=finite
    )


def _inherit(value: _Canonical, nodes: Iterable[_Canonical]) -> _Canonical:
    origins = value.origins.union(*(node.origins for node in nodes))
    return replace(value, origins=origins) if origins != value.origins else value


def _number(state: _MatchingState, value: float | Fraction) -> _Canonical:
    if isinstance(value, float) and not math.isfinite(value):
        return _node(state, "nonfinite", value=repr(value))
    return _node(state, "constant", value=Fraction(value))


def _positive(value: _Canonical) -> bool:
    return (
        value.lower is not None
        and value.lower > 0
        or value.op == "exp"
        and value.arguments[0].finite
    )


def _nonnegative(value: _Canonical) -> bool:
    return value.lower is not None and value.lower >= 0


def _nonzero(value: _Canonical) -> bool:
    return _positive(value) or value.upper is not None and value.upper < 0


def _power(state: _MatchingState, base: _Canonical, exponent: Fraction) -> _Canonical:
    if exponent == 1:
        return base
    if exponent == 0 and _nonzero(base):
        return _inherit(_number(state, 1), (base,))
    if base.op == "constant" and isinstance(base.value, Fraction):
        if (
            exponent.denominator == 1
            and (base.value != 0 or exponent >= 0)
            and abs(exponent.numerator) <= 64
        ):
            return _inherit(_number(state, base.value**exponent.numerator), (base,))
        if exponent == Fraction(1, 2) and base.value >= 0:
            numerator = math.isqrt(base.value.numerator)
            denominator = math.isqrt(base.value.denominator)
            if (
                numerator**2 == base.value.numerator
                and denominator**2 == base.value.denominator
            ):
                return _inherit(
                    _number(state, Fraction(numerator, denominator)), (base,)
                )
    if base.op == "exp" and exponent > 0:
        return _inherit(
            _function(
                state,
                "exp",
                _product(state, (_number(state, exponent), base.arguments[0])),
            ),
            (base,),
        )
    if base.op == "power" and isinstance(base.value, Fraction):
        original = base.arguments[0]
        if base.value == 2 and exponent == Fraction(1, 2):
            return _inherit(_function(state, "abs", original), (base,))
        integers = base.value.denominator == exponent.denominator == 1
        if (integers or _positive(original)) and (
            base.value >= 0 and exponent >= 0 or _nonzero(original)
        ):
            return _inherit(_power(state, original, base.value * exponent), (base,))
    return _node(state, "power", (base,), exponent)


def _coefficient(value: _Canonical) -> tuple[Fraction, _Canonical | None]:
    if value.op == "constant" and isinstance(value.value, Fraction):
        return value.value, None
    if value.op == "mul" and value.arguments[0].op == "constant":
        coefficient = value.arguments[0].value
        if isinstance(coefficient, Fraction):
            return coefficient, value
    return Fraction(1), value


def _sum(state: _MatchingState, values: Iterable[_Canonical]) -> _Canonical:
    pending = list(values)
    terms: list[_Canonical] = []
    removed: list[_Canonical] = []
    while pending:
        value = pending.pop()
        if value.op == "add":
            pending.extend(value.arguments)
            removed.append(value)
        else:
            terms.append(value)
    groups: dict[int, tuple[_Canonical, Fraction]] = {}
    constant = Fraction(0)
    for term in terms:
        coefficient, base = _coefficient(term)
        if base is None:
            constant += coefficient
            removed.append(term)
            continue
        if base.op == "mul" and coefficient != 1:
            base = _product(state, base.arguments[1:], expand=False)
            removed.append(term)
        previous = groups.get(base.key)
        groups[base.key] = base, coefficient + (previous[1] if previous else 0)
    result: list[_Canonical] = []
    for base, coefficient in groups.values():
        if coefficient == 0 and base.finite:
            removed.append(base)
        elif coefficient == 1:
            result.append(base)
        else:
            result.append(
                _product(state, (_number(state, coefficient), base), expand=False)
            )
    if constant != 0:
        result.append(_number(state, constant))
    logarithms = [
        value for value in result if value.op == "log" and _positive(value.arguments[0])
    ]
    if len(logarithms) > 1:
        result = [value for value in result if value not in logarithms]
        result.append(
            _node(
                state,
                "log",
                (_product(state, (value.arguments[0] for value in logarithms)),),
            )
        )
        removed.extend(logarithms)
    # sin(x)^2 + cos(x)^2 is valid for real, finite phases.
    squares: dict[int, dict[str, _Canonical]] = {}
    for term in result:
        if term.op == "power" and term.value == 2:
            function = term.arguments[0]
            if function.op in {"sin", "cos"} and function.arguments[0].finite:
                squares.setdefault(function.arguments[0].key, {})[function.op] = term
    for pair in squares.values():
        if "sin" in pair and "cos" in pair:
            result.remove(pair["sin"])
            result.remove(pair["cos"])
            removed.extend(pair.values())
            result.append(_number(state, 1))
    if not result:
        value = _number(state, 0)
    elif len(result) == 1:
        value = result[0]
    else:
        value = _node(state, "add", tuple(sorted(result, key=lambda item: item.key)))
    return _inherit(value, removed)


def _product(
    state: _MatchingState, values: Iterable[_Canonical], *, expand: bool = True
) -> _Canonical:
    pending = list(values)
    factors: list[_Canonical] = []
    removed: list[_Canonical] = []
    coefficient = Fraction(1)
    while pending:
        value = pending.pop()
        if value.op == "mul":
            pending.extend(value.arguments)
            removed.append(value)
        elif value.op == "constant" and isinstance(value.value, Fraction):
            coefficient *= value.value
            removed.append(value)
        else:
            factors.append(value)
    if coefficient == 0 and all(value.finite for value in factors):
        return _inherit(_number(state, 0), (*removed, *factors))
    additions = [value for value in factors if value.op == "add"]
    count = math.prod(len(value.arguments) for value in additions)
    if expand and additions and count <= 32:
        products: list[list[_Canonical]] = [[_number(state, coefficient)]]
        for value in factors:
            alternatives = value.arguments if value.op == "add" else (value,)
            products = [parts + [part] for parts in products for part in alternatives]
            if value.op == "add":
                removed.append(value)
        return _inherit(
            _sum(state, (_product(state, parts, expand=False) for parts in products)),
            removed,
        )
    grouped: dict[int, tuple[_Canonical, list[tuple[Fraction, _Canonical]]]] = {}
    for value in factors:
        if value.op == "power" and isinstance(value.value, Fraction):
            base, exponent = value.arguments[0], value.value
        else:
            base, exponent = value, Fraction(1)
        grouped.setdefault(base.key, (base, []))[1].append((exponent, value))
    result: list[_Canonical] = []
    for base, entries in grouped.values():
        exponent = sum((item[0] for item in entries), Fraction(0))
        integers = all(item[0].denominator == 1 for item in entries)
        can_merge = (integers or _nonnegative(base)) and (
            all(item[0] >= 0 for item in entries) or _nonzero(base)
        )
        if len(entries) == 1:
            result.append(entries[0][1])
        elif can_merge:
            result.append(_power(state, base, exponent))
            removed.extend(item[1] for item in entries)
        else:
            result.extend(item[1] for item in entries)
    exponentials = [value for value in result if value.op == "exp"]
    if len(exponentials) > 1:
        result = [value for value in result if value.op != "exp"]
        result.append(
            _function(
                state,
                "exp",
                _sum(state, (value.arguments[0] for value in exponentials)),
            )
        )
        removed.extend(exponentials)
    result.sort(key=lambda value: value.key)
    if coefficient != 1 or not result:
        result.insert(0, _number(state, coefficient))
    if len(result) == 1:
        value = result[0]
    else:
        value = _node(state, "mul", tuple(result))
    return _inherit(value, removed)


def _function(state: _MatchingState, op: str, value: _Canonical) -> _Canonical:
    if op == "sqrt":
        return _power(state, value, Fraction(1, 2))
    if op == "abs":
        if _nonnegative(value):
            return value
        if value.op == "abs":
            return value
        if value.op == "mul" and value.arguments[0].op == "constant":
            coefficient = value.arguments[0].value
            if isinstance(coefficient, Fraction) and coefficient < 0:
                return _inherit(
                    _product(
                        state,
                        (
                            _number(state, -coefficient),
                            _function(
                                state,
                                "abs",
                                _product(state, value.arguments[1:], expand=False),
                            ),
                        ),
                    ),
                    (value,),
                )
    if op == "log":
        if value.op == "exp" and value.arguments[0].finite:
            return _inherit(value.arguments[0], (value,))
        if (
            value.op == "power"
            and _positive(value.arguments[0])
            and isinstance(value.value, Fraction)
        ):
            return _inherit(
                _product(
                    state,
                    (
                        _number(state, value.value),
                        _function(state, "log", value.arguments[0]),
                    ),
                ),
                (value,),
            )
    if op == "exp":
        if (
            value.op == "mul"
            and len(value.arguments) == 2
            and value.arguments[0].op == "constant"
            and isinstance(value.arguments[0].value, Fraction)
            and value.arguments[1].op == "log"
            and _positive(value.arguments[1].arguments[0])
        ):
            return _inherit(
                _power(
                    state, value.arguments[1].arguments[0], value.arguments[0].value
                ),
                (value, value.arguments[1]),
            )
        if value.op == "log" and _positive(value.arguments[0]):
            return _inherit(value.arguments[0], (value,))
        if value.op == "add" and all(
            argument.op == "log" and _positive(argument.arguments[0])
            for argument in value.arguments
        ):
            return _inherit(
                _product(
                    state, (argument.arguments[0] for argument in value.arguments)
                ),
                (value, *value.arguments),
            )
    if (
        op in {"sin", "cos", "erf"}
        and value.op == "mul"
        and value.arguments[0].op == "constant"
    ):
        coefficient = value.arguments[0].value
        if isinstance(coefficient, Fraction) and coefficient < 0:
            positive = _product(
                state,
                (_number(state, -coefficient), *value.arguments[1:]),
                expand=False,
            )
            result = _function(state, op, positive)
            return _inherit(
                result
                if op == "cos"
                else _product(state, (_number(state, -1), result)),
                (value,),
            )
    if value.op == "constant" and isinstance(value.value, Fraction):
        number = value.value
        if number == 0 and op in {"sin", "erf", "abs"}:
            return _inherit(_number(state, 0), (value,))
        if number == 0 and op in {"exp", "cos"}:
            return _inherit(_number(state, 1), (value,))
        if number == 1 and op == "log":
            return _inherit(_number(state, 0), (value,))
    return _node(state, op, (value,))
