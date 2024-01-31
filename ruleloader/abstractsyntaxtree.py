"""
Implementation of classes that constitute the abstract syntax tree.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable

from .dp3ifc import DP3Ifc
from .exceptions import SemanticException
from .tokenizer import Token, TokenDebug

if TYPE_CHECKING:
    from conditions import Condition, ConditionFactory


class ASTBase(ABC):
    """Base AST class."""

    def __init__(self):
        self.type: type | None = None
        self.value: Any
        self.debug: TokenDebug

    @abstractmethod
    def to_condition(self, factory: ConditionFactory) -> Condition:
        """Transform self to a Conditon object."""

    @abstractmethod
    def to_eq_comparison(self, factory: ConditionFactory) -> Condition:
        """Transform self to an equality comparison."""

    @abstractmethod
    def get_possible_types(self) -> set[type]:
        """Return typing information about self."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of self."""

    def __repr__(self) -> str:
        return f"[{self.__class__.__name__}|{str(self)}]"


class ASTConstant(ASTBase):
    """AST class for representing constants."""

    def __init__(self, token: Token):
        super().__init__()
        assert token.debug is not None
        self.debug = token.debug
        self.value: Any = None

    def to_condition(self, factory: ConditionFactory) -> Condition:
        return factory.create_condition("constant", self.value)

    def to_eq_comparison(self, factory: ConditionFactory) -> Condition:
        return factory.create_condition("eq", self.value)

    def get_possible_types(self) -> set[type]:
        if self.type == float and int(self.value) == self.value:
            return {int, float}
        assert self.type is not None
        return {self.type}

    def __str__(self) -> str:
        return str(self.value)


class ASTFloat(ASTConstant):
    """AST class for representing float constants."""

    def __init__(self, token: Token) -> None:
        super().__init__(token)
        self.value = float(token.value)
        self.type = float
        assert token.debug is not None
        self.debug = token.debug


class ASTString(ASTConstant):
    """AST class for representing string constants."""

    def __init__(self, token: Token) -> None:
        super().__init__(token)
        self.value = str(token.value)
        self.type = str
        assert token.debug is not None
        self.debug = token.debug


class ASTProperty(ASTBase):
    """AST class for representing property access."""

    def __init__(self, attr: str, debug: TokenDebug) -> None:
        super().__init__()
        self.type = "<unknown due to lack of attribute config>"  # type: ignore
        self.attr = attr
        self.debug = debug

    def __str__(self):
        return f"r.{str(self.attr)}"

    def to_condition(self, factory: ConditionFactory) -> Condition:
        return factory.create_condition("property", self.attr)

    def to_eq_comparison(self, factory: ConditionFactory) -> Condition:
        return factory.create_condition("property", self.attr)

    def get_possible_types(self) -> set[type]:
        return {str, int, float, list[str], list[int], list[float]}

    def get_dp3_types(self, dp3: DP3Ifc) -> set[type] | dict[str, type]:
        """Returns a DP3 configuration type of self.attr."""
        try:
            self.type = dp3.get_attr_converter(self.attr)
        except KeyError as e:
            raise SemanticException(f"Unknown attribute {self.attr}", self.debug.get_loc()) from e
        if isinstance(self.type, type):
            return {self.type}
        return self.type  # type: ignore


class ASTNode(ASTBase):
    """Generic AST node class, assuming two child nodes."""

    def __init__(self, self_token: Token, left_branch: ASTBase, right_branch: ASTBase) -> None:
        super().__init__()
        self.token = self_token
        self.left = left_branch
        self.right = right_branch
        self.debug = left_branch.debug + right_branch.debug

    def __str__(self) -> str:
        return f"({repr(self.left)} {str(self.token)} {repr(self.right)})"

    def to_condition(self, factory: ConditionFactory) -> Condition:
        if not isinstance(self.token, Token) or not factory.is_valid_operator(self.token.value):
            raise SemanticException(f"Cannot make condition from {self}", self.debug.get_loc())
        left = self.left.to_condition(factory)
        right = self.right.to_condition(factory)
        return factory.create_condition(self.token.value, left, right)

    def to_eq_comparison(self, factory: ConditionFactory) -> None:
        raise SemanticException("Attempted to convert an AST node to simple equality comparison", self.debug.get_loc())

    def get_possible_types(self) -> set[type]:
        assert self.type is not None
        return {self.type}


class ASTFunction(ASTBase):
    """AST class representing a function call."""

    def __init__(self, name_token: Token, args: ASTList) -> None:
        super().__init__()
        self.name = name_token
        self.args = args
        self.debug = name_token.debug + args.debug
        self.runtime_arg_type: type | None = None
        self.implementation: Callable[..., Any] | None = None

    def __str__(self):
        return f'{str(self.name.value)}({", ".join(str(x) for x in self.args.list)})'

    def to_condition(self, factory: ConditionFactory) -> Condition:
        args = [x.to_condition(factory) for x in self.args.list]
        return factory.create_condition(self.name.value, args, self.implementation)

    def to_eq_comparison(self, factory: ConditionFactory) -> Condition:
        return self.to_condition(factory)

    def get_possible_types(self) -> set[type]:
        assert self.type is not None
        return {self.type}


class ASTList(ASTBase):
    """AST class representing a list of values."""

    def __init__(self, iterable: Iterable[ASTBase], debug: TokenDebug = None) -> None:
        super().__init__()
        self.list = list(iterable)
        self.complete = False
        self.debug = sum((x.debug for x in self.list), None)  # type: ignore
        if self.debug is not None:
            self.debug += debug
        else:
            self.debug = debug

        for val in self.list:
            if isinstance(val, (ASTList, ASTDict)):
                raise SemanticException("Nested structures are not allowed.", val.debug.get_loc())

    def __str__(self) -> str:
        return str(self.list)

    def to_condition(self, factory: ConditionFactory) -> Condition:
        self.list = [x.to_eq_comparison(factory) for x in self.list]
        return factory.create_condition("tuple_eq", self.list)

    def get_possible_types(self) -> set[type]:
        raise NotImplementedError

    def to_eq_comparison(self, factory: ConditionFactory) -> Condition:
        raise SemanticException("Attempted to convert a tuple eq to simple equality comparison", self.debug.get_loc())


class ASTKeyVal(ASTBase):
    """AST class representing a key:value pair."""

    def __init__(self, first: ASTString, second: ASTBase) -> None:
        super().__init__()
        self.value = first, second
        self.debug = first.debug + second.debug

    def __str__(self) -> str:
        return str(self.value)

    def to_condition(self, factory: ConditionFactory) -> tuple:
        return self.value

    def to_eq_comparison(self, factory: ConditionFactory) -> tuple:
        return self.value

    def get_possible_types(self) -> tuple[str, set[type] | dict[str, type]]:  # type: ignore
        return self.value[0].value, self.value[1].get_possible_types()


class ASTDict(ASTBase):
    """AST class representing a dictionary."""

    def __init__(self, ast_list: ASTList) -> None:
        super().__init__()
        key_val_list = [x.to_condition(None) for x in ast_list.list]
        self.value = dict(key_val_list)
        self.debug = ast_list.debug

        for val in self.value.values():
            if isinstance(val, (ASTDict, ASTList)):
                raise SemanticException("Nested structures are not allowed.", val.debug.get_loc())

    def __str__(self) -> str:
        return str(self.value)

    def to_condition(self, factory: ConditionFactory) -> Condition:
        self.value = {key.to_condition(factory)(None): val.to_eq_comparison(factory) for key, val in self.value.items()}
        return factory.create_condition("dict_eq", self.value)

    def to_eq_comparison(self, factory: ConditionFactory) -> Condition:
        raise SemanticException(
            "Attempted to convert a dictionary comparison to simple equality comparison",
            self.debug.get_loc() if self.debug is not None else None,
        )

    def get_possible_types(self) -> set[type]:
        return set()

    def get_dict_types(self) -> dict[str, set[type]]:
        """Returns a dictionary with sets of types for each specified str key."""
        type_dict = {}
        for key, val in self.value.items():
            if isinstance(val, ASTFunction):
                assert val.runtime_arg_type is not None
                type_dict[key.value] = {val.runtime_arg_type}
                continue
            type_dict[key.value] = {val.type}
            if val.type == float and int(val.value) == val.value:
                type_dict[key.value] = {int, float}
        return type_dict
