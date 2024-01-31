"""
Definition of exception types used in RuleLoader implementation.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenizer import Token, Tokens


class RuleLoaderException(Exception):
    """Base for all exceptions raised by the `RuleLoader` class."""

    def __init__(self, message: str, faulty_loc: str = None) -> None:
        self.faulty_loc = faulty_loc
        self.original_args: Any = message
        if faulty_loc is None:
            faulty_loc = "[Infromation for finding faulty LOC missing]"
        super().__init__(f"{message}\nError ocurred here:\n{faulty_loc}")


class TokenizerException(RuleLoaderException):
    """A tokenizer / lexical error exception class."""

    def __init__(self, pos: int, text_len: int, faulty_loc: str = None) -> None:
        super().__init__(f"Tokenizer Error: Unrecognized character at pos {pos} of {text_len}.\n", faulty_loc)
        self.original_args = [pos, text_len]


class SyntaxException(RuleLoaderException):
    """A syntax error exception class."""

    def __init__(self, token: Token, expected_type: Tokens, faulty_loc: str = None) -> None:
        super().__init__(
            f"Syntax Error: Unexpected token {str(token)} while parsing config, "
            f"expected type was <{expected_type.name}>",
            faulty_loc,
        )
        self.original_args = [token, expected_type]


class SyntaxManyException(RuleLoaderException):
    """A syntax error exception class when expecting one of multiple tokens."""

    def __init__(self, token: Token, expected_types: list[Tokens], faulty_loc: str = None) -> None:
        type_names = [f"<{token_type.name}>" for token_type in expected_types]
        super().__init__(
            f"Syntax Error: Unexpected token {str(token)} while parsing config,"
            f' expected one of: {", ".join(type_names)}',
            faulty_loc,
        )
        self.original_args = [token, expected_types]


class PrecedenceException(RuleLoaderException):
    """A syntax error exception class when processing expressions."""

    def __init__(self, token: Token, top_term: Token, faulty_loc: str = None) -> None:
        super().__init__(
            f"Syntax Error: Unexpected token {str(token)} while parsing expression, "
            f"top term on stack was {str(top_term)}",
            faulty_loc,
        )
        self.original_args = [token, top_term]


class PrecedenceMissingRuleException(RuleLoaderException):
    """A syntax error exception when processing expressions and the parsed sequence makes no sense."""

    def __init__(self, handle: tuple, faulty_loc: str = None) -> None:
        super().__init__(
            f"Syntax Error: Unable to meaningfully incorporate the sequence {handle} into the expression\n", faulty_loc
        )
        self.original_args = [handle, faulty_loc]


class SemanticException(RuleLoaderException):
    """A semantic error exception class."""

    def __init__(self, message: str, faulty_loc: str = None) -> None:
        super().__init__(f"Semantic Error: {message}", faulty_loc)
        self.original_args = [message]
