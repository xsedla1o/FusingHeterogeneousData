"""
Implementation of the Tokenizer class and related classes.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from __future__ import annotations

import re
from collections.abc import Generator, Iterator
from enum import Enum
from typing import Any

from ruleloader.exceptions import TokenizerException


class Tokens(Enum):
    """An Enum class that identifies all accepted tokens."""

    word = "word"
    property = "property"
    float = "float"
    string = "string"
    open_brack = "open_brack"
    close_brack = "close_brack"
    open_curly = "open_curly"
    close_curly = "close_curly"
    open_square = "open_square"
    close_square = "close_square"
    colon = "colon"
    comma = "comma"
    lt = "lt"
    gt = "gt"
    eq = "eq"
    dash = "dash"
    newline = "newline"
    whitespace = "whitespace"
    eof = "EOF"
    neg = "neg"
    precedence_beg_handle = "<"
    comment = "comment"


class TokenDebug:
    """Class to represent token location in the source code."""

    __slots__ = "source_line", "begin_pos", "end_pos"

    def __init__(self, source_line: tuple[int, str], begin_pos: int, end_pos: int) -> None:
        self.source_line = source_line
        self.begin_pos = begin_pos
        self.end_pos = end_pos

    def get_loc(self) -> str:
        segment_length = self.end_pos - self.begin_pos
        loc_num, text = self.source_line
        return f"{loc_num + 1:3d}> {text}\n" + " " * (5 + self.begin_pos) + "^" * segment_length

    def __add__(self, other: TokenDebug | None) -> TokenDebug:
        if other is None:
            return self
        if isinstance(other, TokenDebug):
            if self.begin_pos < other.begin_pos:
                new_beg = self.begin_pos
                new_end = other.end_pos
            else:
                new_beg = other.begin_pos
                new_end = self.end_pos
            return TokenDebug(self.source_line, new_beg, new_end)
        raise TypeError("TokenDebug supports addition only with itself and with None.")

    def __radd__(self, other: TokenDebug | None) -> TokenDebug:
        return self + other


class Token:
    """Class to represent a single token"""

    __slots__ = ["type", "value", "annot", "debug"]

    def __init__(self, token_type: Any, value: str, annot=None, debug: TokenDebug = None) -> None:
        self.type: Any = token_type
        self.value: Any = value
        self.annot: str = annot
        self.debug: TokenDebug | None = debug

    def __str__(self) -> str:
        return f"<{self.type.name}: {self.value.__repr__()}>"

    def __repr__(self) -> str:
        if self.annot is not None:
            return f"<{self.annot.__repr__()}>"
        if isinstance(self.type, Enum):
            return str(self)
        return f"<{self.type.__repr__()}: {self.value.__repr__()}>"


class Tokenizer:
    """Class representing a token parser."""

    token_pattern = r"""
    (?P<word>[a-zA-Z_][a-zA-Z0-9_]*)
    |(?P<property>\.[a-zA-Z0-9_]+)
    |(?P<float>-?\d+(?:.\d+)?)
    |(?P<string1>'([^\n']*)')    # simple quotes
    |(?P<string2>\"([^\n\"]*)\") # double quotes
    |(?P<open_brack>\()
    |(?P<close_brack>\))
    |(?P<open_curly>{)
    |(?P<close_curly>})
    |(?P<open_square>\[)
    |(?P<close_square>\])
    |(?P<colon>:)
    |(?P<comma>,)
    |(?P<lt>>)
    |(?P<gt><)
    |(?P<eq>==)
    |(?P<dash>-)
    |(?P<newline>\n)
    |(?P<neg>\!)
    |(?P<comment>[\t ]*\#[^\n]*\n?)
    |(?P<whitespace>[\t ]+)
    """
    token_re = re.compile(token_pattern, re.VERBOSE)

    def __init__(self, source: str) -> None:
        """Init the token parser."""
        self.source_lines = source.split("\n")
        self.line_iter: Iterator[tuple[int, str]]
        self.curr_line: tuple[int, str] | None = None
        self.line_pos = 0

        self.eof_token = Token(Tokens("EOF"), "EOF")
        self.tokens = list(self.tokenize(source))

        # Contextually remove comment tokens
        i = 0
        while i < len(self.tokens):
            if self.tokens[i].type == Tokens.comment:
                if i < 1 or self.tokens[i - 1].type == Tokens.newline:
                    self.tokens.pop(i)
                    continue
                self.tokens[i].type = Tokens.newline
            i += 1

        self.token_iterator: Iterator[Token]
        self.iter_index = -1

    def tokenize(self, text: str) -> Generator[Token, None, None]:
        """Turn passed text into a list of tokens."""
        self.line_iter = iter(enumerate(self.source_lines))
        self.curr_line = next(self.line_iter, self.curr_line)
        assert self.curr_line is not None
        self.line_pos = 0

        pos = 0
        while True:
            m = self.token_re.match(text, pos)
            if not m:
                self.eof_token.debug = TokenDebug(self.curr_line, pos - 1, pos)
                break
            pos = m.end()
            if m.lastgroup == "comment":
                yield Token(
                    Tokens("comment"),
                    "\n",
                    debug=TokenDebug(self.curr_line, m.start() - self.line_pos, m.end() - self.line_pos),
                )
                self.curr_line = next(self.line_iter, self.curr_line)
                assert self.curr_line is not None
                self.line_pos = pos
                continue
            token = self.create_token_string_safe(m)
            if token.type == Tokens.newline:
                self.curr_line = next(self.line_iter, self.curr_line)
                assert self.curr_line is not None
                self.line_pos = pos
            yield token
        if pos != len(text):
            raise TokenizerException(pos, len(text), self.curr_line[1])

    def create_token_string_safe(self, m: re.Match) -> Token:
        """Create a token based on a re.Match object."""
        assert self.curr_line is not None
        debug = TokenDebug(self.curr_line, m.start() - self.line_pos, m.end() - self.line_pos)
        if m.lastgroup in ("string1", "string2"):
            value = m.group(m.lastgroup)[1:-1]
            return Token(Tokens("string"), value, debug=debug)
        return Token(Tokens(m.lastgroup), m.group(m.lastgroup), debug=debug)  # type: ignore

    def seek(self, iterator_offset: int) -> Token:
        """Seek a token that is `iterator_offset` away from current iterator position."""
        index = self.iter_index + iterator_offset
        if index < 0:
            return self.tokens[0]
        if index >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.iter_index + iterator_offset]

    def __iter__(self) -> Tokenizer:
        self.token_iterator = iter(self.tokens)
        self.iter_index = -1
        self.line_iter = iter(enumerate(self.source_lines))
        self.curr_line = next(self.line_iter, self.curr_line)
        return self

    def __next__(self) -> Token:
        self.iter_index += 1
        next_token = next(self.token_iterator, self.eof_token)
        if next_token.type == Tokens.newline:
            self.curr_line = next(self.line_iter, self.curr_line)
        return next_token
