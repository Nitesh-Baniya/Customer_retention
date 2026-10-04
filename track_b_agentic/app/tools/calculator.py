from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import RegisteredTool

Number = int | float
MathFunction = Callable[..., Number]


def _factorial(value: Number) -> int:
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("factorial requires a whole number")

    integer = int(value)

    if integer < 0 or integer > 170:
        raise ValueError("factorial input must be between 0 and 170")

    return math.factorial(integer)


def _rounded(value: Number, decimal_places: Number = 0) -> Number:
    if isinstance(decimal_places, float) and not decimal_places.is_integer():
        raise ValueError("round decimal places must be a whole number")

    places = int(decimal_places)

    if abs(places) > 15:
        raise ValueError("round decimal places must be between -15 and 15")

    return round(value, places)


def _percentage_of(percent: Number, value: Number) -> float:
    return (percent / 100) * value


def _root(value: Number, degree: Number = 2) -> float:
    if degree == 0:
        raise ValueError("root degree cannot be zero")

    if isinstance(degree, float) and not degree.is_integer():
        raise ValueError("root degree must be a whole number")

    degree_int = int(degree)

    if value < 0:
        if degree_int % 2 == 0:
            raise ValueError("even root of a negative number is not real")

        return -((-value) ** (1 / degree_int))

    return value ** (1 / degree_int)


def _mean(*values: Number) -> float:
    if not values:
        raise ValueError("mean requires at least one value")

    return sum(values) / len(values)


def _median(*values: Number) -> float:
    if not values:
        raise ValueError("median requires at least one value")

    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2

    if count % 2 == 1:
        return float(ordered[middle])

    return (ordered[middle - 1] + ordered[middle]) / 2


def _variance(*values: Number) -> float:
    if not values:
        raise ValueError("variance requires at least one value")

    average = _mean(*values)

    return sum((value - average) ** 2 for value in values) / len(values)


def _stddev(*values: Number) -> float:
    return math.sqrt(_variance(*values))


def _combination(n: Number, r: Number) -> int:
    if isinstance(n, float) and not n.is_integer():
        raise ValueError("combination requires whole numbers")

    if isinstance(r, float) and not r.is_integer():
        raise ValueError("combination requires whole numbers")

    n_int = int(n)
    r_int = int(r)

    if n_int < 0:
        raise ValueError("n must be non-negative")

    if r_int < 0 or r_int > n_int:
        raise ValueError("combination requires 0 <= r <= n")

    return math.comb(n_int, r_int)


def _permutation(n: Number, r: Number) -> int:
    if isinstance(n, float) and not n.is_integer():
        raise ValueError("permutation requires whole numbers")

    if isinstance(r, float) and not r.is_integer():
        raise ValueError("permutation requires whole numbers")

    n_int = int(n)
    r_int = int(r)

    if n_int < 0:
        raise ValueError("n must be non-negative")

    if r_int < 0 or r_int > n_int:
        raise ValueError("permutation requires 0 <= r <= n")

    return math.perm(n_int, r_int)


def _simple_interest(
    principal: Number,
    rate_percent: Number,
    time: Number,
) -> float:
    if principal < 0:
        raise ValueError("principal cannot be negative")

    if time < 0:
        raise ValueError("time cannot be negative")

    return (principal * rate_percent * time) / 100


def _compound_interest(
    principal: Number,
    rate_percent: Number,
    time: Number,
    compounds_per_period: Number = 1,
) -> float:
    if principal < 0:
        raise ValueError("principal cannot be negative")

    if time < 0:
        raise ValueError("time cannot be negative")

    if compounds_per_period <= 0:
        raise ValueError("compounds_per_period must be positive")

    if (
        isinstance(compounds_per_period, float)
        and not compounds_per_period.is_integer()
    ):
        raise ValueError("compounds_per_period must be a whole number")

    compounds = int(compounds_per_period)

    amount = principal * (1 + rate_percent / (100 * compounds)) ** (compounds * time)

    return amount - principal


def _sin_deg(value: Number) -> float:
    return math.sin(math.radians(value))


def _cos_deg(value: Number) -> float:
    return math.cos(math.radians(value))


def _tan_deg(value: Number) -> float:
    return math.tan(math.radians(value))


def _asin_deg(value: Number) -> float:
    return math.degrees(math.asin(value))


def _acos_deg(value: Number) -> float:
    return math.degrees(math.acos(value))


def _atan_deg(value: Number) -> float:
    return math.degrees(math.atan(value))


def _gcd(a: Number, b: Number) -> int:
    if isinstance(a, float) and not a.is_integer():
        raise ValueError("gcd requires whole numbers")

    if isinstance(b, float) and not b.is_integer():
        raise ValueError("gcd requires whole numbers")

    return math.gcd(int(a), int(b))


def _lcm(a: Number, b: Number) -> int:
    if isinstance(a, float) and not a.is_integer():
        raise ValueError("lcm requires whole numbers")

    if isinstance(b, float) and not b.is_integer():
        raise ValueError("lcm requires whole numbers")

    return math.lcm(int(a), int(b))


def _hypotenuse(a: Number, b: Number) -> float:
    return math.hypot(a, b)


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "A safe mathematical expression using supported operators, constants, "
            "and functions. Use +, -, *, /, //, %, and ^ or ** for powers. "
            "Supported constants: pi, e, tau. "
            "Supported functions include sqrt(x), root(x, n), abs(x), round(x, n), "
            "ceil(x), floor(x), log(x), log(x, base), log10(x), "
            "sin(x), cos(x), tan(x), asin(x), acos(x), atan(x), "
            "sin_deg(x), cos_deg(x), tan_deg(x), asin_deg(x), acos_deg(x), atan_deg(x), "
            "degrees(x), radians(x), factorial(x), comb(n, r), perm(n, r), "
            "gcd(a, b), lcm(a, b), hypot(a, b), percentage_of(percent, value), "
            "mean(...), median(...), variance(...), stddev(...), min(...), max(...), "
            "simple_interest(principal, rate_percent, time), and "
            "compound_interest(principal, rate_percent, time, compounds_per_period). "
            "For cube roots, use root(x, 3). "
            "Examples: sqrt(81), root(125, 3), 2^8, percentage_of(15, 200), "
            "mean(10, 20, 30), comb(10, 3), sin_deg(30), "
            "compound_interest(100000, 10, 2)."
        ),
    )


class SafeCalculator:
    """Evaluate arithmetic expressions without executing arbitrary Python."""

    MAX_AST_NODES = 80
    MAX_ABSOLUTE_RESULT = 1e15

    BINARY_OPERATORS: ClassVar[dict[type[ast.operator], MathFunction]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    UNARY_OPERATORS: ClassVar[dict[type[ast.unaryop], MathFunction]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    CONSTANTS: ClassVar[dict[str, Number]] = {
        "e": math.e,
        "pi": math.pi,
        "tau": math.tau,
    }

    # Each entry stores:
    # function, minimum arguments, maximum arguments.
    FUNCTIONS: ClassVar[dict[str, tuple[MathFunction, int, int]]] = {
        "abs": (abs, 1, 1),
        "acos": (math.acos, 1, 1),
        "acos_deg": (_acos_deg, 1, 1),
        "asin": (math.asin, 1, 1),
        "asin_deg": (_asin_deg, 1, 1),
        "atan": (math.atan, 1, 1),
        "atan_deg": (_atan_deg, 1, 1),
        "ceil": (math.ceil, 1, 1),
        "comb": (_combination, 2, 2),
        "compound_interest": (_compound_interest, 3, 4),
        "cos": (math.cos, 1, 1),
        "cos_deg": (_cos_deg, 1, 1),
        "degrees": (math.degrees, 1, 1),
        "factorial": (_factorial, 1, 1),
        "floor": (math.floor, 1, 1),
        "gcd": (_gcd, 2, 2),
        "hypot": (_hypotenuse, 2, 2),
        "lcm": (_lcm, 2, 2),
        "log": (math.log, 1, 2),
        "log10": (math.log10, 1, 1),
        "max": (max, 2, 10),
        "mean": (_mean, 1, 10),
        "median": (_median, 1, 10),
        "min": (min, 2, 10),
        "percentage_of": (_percentage_of, 2, 2),
        "perm": (_permutation, 2, 2),
        "radians": (math.radians, 1, 1),
        "root": (_root, 1, 2),
        "round": (_rounded, 1, 2),
        "simple_interest": (_simple_interest, 3, 3),
        "sin": (math.sin, 1, 1),
        "sin_deg": (_sin_deg, 1, 1),
        "sqrt": (math.sqrt, 1, 1),
        "stddev": (_stddev, 1, 10),
        "tan": (math.tan, 1, 1),
        "tan_deg": (_tan_deg, 1, 1),
        "variance": (_variance, 1, 10),
    }

    def evaluate(self, expression: str) -> Number:
        # Allow calculator-style ^ for exponentiation.
        expression = expression.replace("^", "**")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid mathematical expression") from exc

        if sum(1 for _ in ast.walk(tree)) > self.MAX_AST_NODES:
            raise ValueError("Expression is too complex")

        try:
            result = self._evaluate_node(tree.body)
        except ZeroDivisionError as exc:
            raise ValueError("Division by zero is not allowed") from exc
        except OverflowError as exc:
            raise ValueError("Calculation overflowed") from exc
        except ValueError:
            raise
        except ArithmeticError as exc:
            raise ValueError("Invalid mathematical operation") from exc

        self._validate_result(result)
        return result

    def _evaluate_node(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Constant):
            return self._read_number(node.value)

        if isinstance(node, ast.Name):
            return self._read_constant(node.id)

        if isinstance(node, ast.BinOp):
            if type(node.op) not in self.BINARY_OPERATORS:
                raise ValueError("Binary operator is not supported")

            return self._evaluate_binary(node)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.UNARY_OPERATORS:
                raise ValueError("Unary operator is not supported")

            return self._evaluate_unary(node)

        if isinstance(node, ast.Call):
            return self._evaluate_function(node)

        raise ValueError("Expression contains an unsupported operation")

    @staticmethod
    def _read_number(value: object) -> Number:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only numeric values are allowed")

        return value

    def _read_constant(self, name: str) -> Number:
        if name not in self.CONSTANTS:
            raise ValueError(f"Unknown constant: {name}")

        return self.CONSTANTS[name]

    def _evaluate_binary(self, node: ast.BinOp) -> Number:
        left = self._evaluate_node(node.left)
        right = self._evaluate_node(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > 10:
                raise ValueError("Exponent is too large")

            if abs(left) > 1e8:
                raise ValueError("Base is too large for exponentiation")

        operation = self.BINARY_OPERATORS[type(node.op)]

        return operation(left, right)

    def _evaluate_unary(self, node: ast.UnaryOp) -> Number:
        operation = self.UNARY_OPERATORS[type(node.op)]

        return operation(self._evaluate_node(node.operand))

    def _evaluate_function(self, node: ast.Call) -> Number:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")

        if node.keywords:
            raise ValueError("Keyword arguments are not allowed")

        function_spec = self.FUNCTIONS.get(node.func.id)

        if function_spec is None:
            raise ValueError(f"Unknown function: {node.func.id}")

        function, minimum_args, maximum_args = function_spec

        if not minimum_args <= len(node.args) <= maximum_args:
            if minimum_args == maximum_args:
                raise ValueError(
                    f"{node.func.id} expects exactly {minimum_args} argument"
                    f"{'s' if minimum_args != 1 else ''}"
                )

            raise ValueError(
                f"{node.func.id} expects between "
                f"{minimum_args} and {maximum_args} arguments"
            )

        arguments = [self._evaluate_node(argument) for argument in node.args]

        result = function(*arguments)

        self._validate_result(result)

        return result

    def _validate_result(self, result: Number) -> None:
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError("Result must be numeric")

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Result must be finite")

        if abs(result) > self.MAX_ABSOLUTE_RESULT:
            raise ValueError("Result is too large")


def calculate(input_data: BaseModel) -> str:
    calculator_input = CalculatorInput.model_validate(input_data.model_dump())

    calculator = SafeCalculator()

    result = calculator.evaluate(calculator_input.expression)

    return json.dumps(
        {
            "expression": calculator_input.expression,
            "result": result,
        }
    )


def create_calculator_tool() -> RegisteredTool:
    return RegisteredTool(
        name="calculator",
        description=(
            "Use this tool for reliable mathematical calculations instead of doing "
            "arithmetic mentally. It supports arithmetic, powers, roots, logarithms, "
            "percentages, trigonometry, statistics, factorials, permutations, "
            "combinations, GCD, LCM, geometry, and simple/compound interest. "
            "Use sqrt(x) for square roots and root(x, n) for nth roots, including "
            "root(x, 3) for cube roots. Standard sin/cos/tan use radians; use "
            "sin_deg/cos_deg/tan_deg for degree inputs. Use only documented function "
            "names and do not invent functions such as cbrt unless explicitly supported."
        ),
        input_model=CalculatorInput,
        handler=calculate,
    )
