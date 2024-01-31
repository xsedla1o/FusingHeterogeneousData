"""
Implementation of classes that constitute rule conditions.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable

from .exceptions import SemanticException


def aggregate_if_list(first, second):
    """If both are lists, return their union, else return first."""
    if isinstance(first, list) and isinstance(second, list):
        return [*first, *[item for item in second if item not in first]]
    return first


class Condition(ABC):
    """Abstract base class for all conditions."""

    __slots__ = ()
    op = ""

    @classmethod
    def get_subclass_factory_dict(cls) -> dict[str, type[Condition]]:
        """Return a dict containing <operator>:<class to construct> pairs."""
        return {operator.op: operator for operator in cls.__subclasses__()}

    def __call__(self, record: dict):
        return self.eval(record)

    @abstractmethod
    def eval(self, record: dict):
        """The eval method contains the functionality of the condition node."""

    @abstractmethod
    def get_attr_watchset(self):
        """Return a set of attributes the condition depends on."""

    @abstractmethod
    def get_trigger_dictionary(self, record: dict) -> dict:
        """Return a dict of all the record's attributes that triggered the condition."""

    @staticmethod
    def update_trigger_dictionary(to_update: dict, to_update_with: dict):
        """Updates to_update with values of to_update_with."""
        for key, value in to_update_with.items():
            if key in to_update:
                to_update[key] = aggregate_if_list(to_update[key], value)
            else:
                to_update[key] = value


class LogicOperator(Condition):
    """Base for logic operators."""

    __slots__ = "left", "right"

    def __init__(self, left: Condition, right: Condition) -> None:
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"

    @abstractmethod
    def eval(self, record: dict):
        """The eval method contains the functionality of the logical node."""

    def get_attr_watchset(self):
        return self.left.get_attr_watchset() | self.right.get_attr_watchset()


class And(LogicOperator):
    """Conjunction operator representation."""

    __slots__ = ()
    op = "and"

    def eval(self, record: dict) -> bool:
        return self.left(record) and self.right(record)

    def get_trigger_dictionary(self, record: dict) -> dict:
        left_trigger = self.left.get_trigger_dictionary(record)
        right_trigger = self.right.get_trigger_dictionary(record)
        self.update_trigger_dictionary(left_trigger, right_trigger)
        return left_trigger


class Or(LogicOperator):
    """Disjunction operator representation."""

    __slots__ = ()
    op = "or"

    def eval(self, record: dict) -> bool:
        return self.left(record) or self.right(record)

    def get_trigger_dictionary(self, record: dict) -> dict:
        if self.left(record):
            return self.left.get_trigger_dictionary(record)
        return self.right.get_trigger_dictionary(record)


class ComparisonOperator(Condition):
    """Base for comparison operators."""

    __slots__ = "left", "right"

    def __init__(self, left: Condition, right: Condition) -> None:
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"

    @abstractmethod
    def eval(self, record: dict):
        """The eval method contains the functionality of the comparison node."""

    def get_attr_watchset(self):
        return self.left.get_attr_watchset() | self.right.get_attr_watchset()

    def get_trigger_dictionary(self, record: dict) -> dict:
        left_trigger = self.left.get_trigger_dictionary(record)
        right_trigger = self.right.get_trigger_dictionary(record)
        self.update_trigger_dictionary(left_trigger, right_trigger)
        return left_trigger


class LessThan(ComparisonOperator):
    """True if left arg is smaller than the right."""

    __slots__ = ()
    op = "<"

    def eval(self, record: dict) -> bool:
        return self.left(record) < self.right(record)


class GreaterThan(ComparisonOperator):
    """True if left arg is greater than the right."""

    __slots__ = ()
    op = ">"

    def eval(self, record: dict) -> bool:
        return self.left(record) > self.right(record)


class Equals(ComparisonOperator):
    """Equality operator."""

    __slots__ = ()
    op = "=="

    def eval(self, record: dict) -> bool:
        return self.left(record) == self.right(record)


class In(ComparisonOperator):
    """
    Checks whether the left arg is in the right arg.

    Allows for structural pattern-matching, see docs (README) for more.
    """

    __slots__ = ()
    op = "in"

    def __init__(self, left: Condition, right: Condition) -> None:
        super().__init__(left, right)
        if isinstance(self.left, Constant):
            self.left = SimpleEqualityCheck(left({}))

    def eval(self, record: dict) -> bool:
        if self.right(record) is None:
            return False
        return any(self.left(r) for r in self.right(record))

    def get_trigger_dictionary(self, record: dict) -> dict:
        for item in self.right(record):
            if self.left(item):
                left_trigger = self.left.get_trigger_dictionary(item)
                right_trigger = self.right.get_trigger_dictionary(record)
                if len(right_trigger) > 1:
                    raise Exception
                for key in right_trigger:
                    return {key: [left_trigger[None]]}
        return {}

    def get_attr_watchset(self):
        return self.right.get_attr_watchset()


class Function(Condition):
    """Abstract base for all functions."""

    __slots__ = "original_args", "args", "type", "input", "implementation", "last_trigger"
    name = ""

    def __init__(self, args: list, function_implementation: Callable) -> None:
        self.original_args = args
        # self.args = args
        if not isinstance(args, list):
            raise SemanticException(f"Function {self.__class__} expected a list of arguments, but recieved {args}")
        self.implementation = function_implementation
        self.last_trigger: dict | None = None

    def __str__(self) -> str:
        return (
            f"{self.name}("
            + ", ".join([str(arg) if not isinstance(arg, str) else repr(arg) for arg in self.original_args])
            + ")"
        )

    def __repr__(self) -> str:
        return str(self)

    def eval(self, record: dict):
        return self.implementation(self, record)

    def get_trigger_dictionary(self, record: dict) -> dict:
        self.implementation(self, record)
        assert self.last_trigger is not None
        return self.last_trigger

    @classmethod
    @abstractmethod
    def get_signature(cls) -> list[tuple[Callable, tuple, type | None, type]]:
        """Returns list of possible signatures: (implementing_fn, arg_types_tuple, runtime_arg_type, return_type)."""

    @staticmethod
    def signature_to_str(signature: tuple[tuple, type | None, type]) -> str:
        """Returns a string with the function signature."""

        def to_str(val):
            return val.__name__ if isinstance(val, type) else str(val)

        compiletime_args, runtime_arg, ret_type = signature
        if runtime_arg is None:
            return f'({", ".join(to_str(x) for x in compiletime_args)}) -> () -> {ret_type.__name__}'
        return f'({", ".join(to_str(x) for x in compiletime_args)}) -> ({runtime_arg.__name__}) -> {ret_type.__name__}'

    @classmethod
    def get_subclass_factory_dict(cls) -> dict[str, type[Condition]]:
        return {function.name: function for function in cls.__subclasses__()}


class Contains(Function):
    """Provides a 'substring' in 'string' functionality."""

    __slots__ = ()
    name = "contains"

    def __init__(self, args: list, function_implementation: Callable) -> None:
        super().__init__(args, function_implementation)
        if self.implementation == Contains.standalone:
            self.input = args[1]
        self.args: Condition = args[0]

    @classmethod
    def get_signature(cls) -> list[tuple[Callable, tuple, type | None, type]]:
        return [
            (Contains.inside_expr, (str,), str, bool),
            (Contains.standalone, (str, str), None, bool),
        ]

    def standalone(self, record: dict) -> bool:
        """Standalone implementation."""
        if self.input(record) is None:
            return False
        self.last_trigger = self.input.get_trigger_dictionary(record)
        return self.args(record) in self.input(record)

    def inside_expr(self, record: dict) -> bool:
        """Implementation when inside expression."""
        if record is None:
            return False
        self.last_trigger = {None: record}
        return self.args(record) in record

    def get_attr_watchset(self):
        if self.implementation == self.standalone:  # pylint: disable=comparison-with-callable
            return self.input.get_attr_watchset()
        return set()


class Re(Function):
    """Interface for the re module.

    The first argument will be passed to re.compile,
    the second or runtime argument will be compared
    with the resulting pattern with re.match."""

    __slots__ = ("pattern",)
    name = "re"

    def __init__(self, args: list, function_implementation: Callable) -> None:
        super().__init__(args, function_implementation)
        pattern = args[0](None)
        self.pattern = re.compile(pattern)

        if self.implementation == Re.standalone:
            self.input = args[1]

    @classmethod
    def get_signature(cls) -> list[tuple[Callable, tuple, type | None, type]]:
        return [
            (Re.inside_expr, (str,), str, bool),
            (Re.standalone, (str, str), None, bool),
        ]

    def inside_expr(self, record: str) -> bool:
        """Implementation inside expression."""
        if record is None:
            return False
        self.last_trigger = {None: record}
        return bool(re.search(self.pattern, record))

    def standalone(self, record: str) -> bool:
        """Standalone implementation."""
        if self.input(record) is None:
            return False
        self.last_trigger = self.input.get_trigger_dictionary(record)
        return bool(re.search(self.pattern, self.input(record)))

    def get_attr_watchset(self):
        if self.implementation == self.standalone:  # pylint: disable=comparison-with-callable
            return self.input.get_attr_watchset()
        return set()


class ContainsAnyOf(Function):
    """Accepts any number of string arguments and returns whether any of them is a substring to the runtime argument."""

    __slots__ = ()
    name = "contains_any_of"

    def __init__(self, args: list, function_implementation: Callable) -> None:
        super().__init__(args, function_implementation)
        self.args = [x(None) for x in self.original_args]

    @classmethod
    def get_signature(cls) -> list[tuple[Callable, tuple, type | None, type]]:
        return [(ContainsAnyOf.inside_expr, (str, ...), str, bool)]

    def inside_expr(self, record: dict) -> bool:
        """Implementation inside expression."""
        if record is None:
            return False
        self.last_trigger = {None: record}
        return any(substring in record for substring in self.args)

    def get_attr_watchset(self):
        return set()


class Len(Function):
    """Interface for builtin len function.
    It is currently supported for strings and arrays of primitive types."""

    __slots__ = ()
    name = "len"

    def __init__(self, args: list, function_implementation: Callable) -> None:
        super().__init__(args, function_implementation)
        self.args = args[0]

    @classmethod
    def get_signature(cls) -> list[tuple[Callable, tuple, type | None, type]]:
        return [
            (Len.inside_expr, (str,), None, int),
            (Len.inside_expr, (list[str],), None, int),
            (Len.inside_expr, (list[float],), None, int),
            (Len.inside_expr, (list[int],), None, int),
        ]

    def inside_expr(self, record: dict) -> int:
        """Implementation inside expression."""
        value = self.args(record)
        self.last_trigger = self.args.get_trigger_dictionary(record)
        if value is not None:
            return len(value)
        return -1

    def get_attr_watchset(self):
        return set()


class EqualityCheck(Condition):
    """Base for different equality implementations for different types."""

    __slots__ = ("value",)
    name = ""

    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def eval(self, record: dict):
        """The eval method contains the functionality of the equality check node."""

    @classmethod
    def get_subclass_factory_dict(cls) -> dict[str, type[Condition]]:
        return {check.name: check for check in cls.__subclasses__()}

    def get_trigger_dictionary(self, record: dict) -> dict:
        if self.eval(record):
            return {None: record}
        return {}

    def get_attr_watchset(self):
        return set()


class SimpleEqualityCheck(EqualityCheck):
    """Equality of primitive types."""

    __slots__ = ()
    name = "eq"

    def eval(self, record: dict) -> bool:
        return self.value == record


class TupleEquailityCheck(EqualityCheck):
    """Equality of tuples."""

    __slots__ = ()
    name = "tuple_eq"

    def eval(self, record) -> bool:
        if not isinstance(record, tuple) or len(self.value) != len(record):
            return False
        for index, item in enumerate(self.value):
            if not item(record[index]):
                return False
        return True


class DictEquailityCheck(EqualityCheck):
    """Equality of dictionaries."""

    __slots__ = ()
    name = "dict_eq"

    def eval(self, record: dict) -> bool:
        if record is None:
            return False
        for key, val in self.value.items():
            if key not in record:
                return False
            if not val(record[key]):
                return False
        return True


class PropertyAccess(Condition):
    """Returns value of property."""

    __slots__ = ("attr",)

    def __init__(self, attr: str) -> None:
        self.attr = attr

    def __str__(self) -> str:
        return f".{self.attr}"

    def __repr__(self) -> str:
        return f"[PropAccess|{self}]"

    def eval(self, record: dict):
        if self.attr in record:
            return record[self.attr]
        return None

    def get_trigger_dictionary(self, record: dict) -> dict:
        if self.attr in record:
            return {self.attr: record[self.attr]}
        return {}

    @classmethod
    def get_subclass_factory_dict(cls) -> dict[str, type[Condition]]:
        return {"property": PropertyAccess}

    def get_attr_watchset(self):
        return {self.attr}


class Constant(Condition):
    """Represents a constant value."""

    __slots__ = ("value",)

    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)

    def eval(self, record):
        return self.value

    def get_trigger_dictionary(self, record: dict) -> dict:
        return {}

    @classmethod
    def get_subclass_factory_dict(cls) -> dict[str, type[Condition]]:
        return {"constant": Constant}

    def get_attr_watchset(self):
        return set()


class ConditionFactory:
    """Creates `Condition` objects based on their name, see `create_condition`."""

    def __init__(self) -> None:
        self.condition_database = {}
        for c in Condition.__subclasses__():
            self.condition_database.update(c.get_subclass_factory_dict())

    def is_valid_operator(self, operator: str) -> bool:
        return operator in self.condition_database

    def is_valid_function(self, name: str) -> bool:
        if name in self.condition_database:
            return issubclass(self.condition_database[name], Function)
        return False

    def get_function_signatures(self, name: str) -> list[tuple[Callable, tuple, type | None, type]]:
        return self.condition_database[name].get_signature()  # type: ignore

    def create_condition(self, operator: str, *args) -> Condition:
        """Creates `Condition` objects based on passed `operator`,
        all other args are passed to the initialized Condition."""
        if operator not in self.condition_database:
            raise SemanticException(
                f"Cannot make condition from operator '{operator}'\n"
                f"Called with args: {' '.join(str(x) for x in args)}"
            )
        return self.condition_database[operator](*args)
