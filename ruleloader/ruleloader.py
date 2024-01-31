"""
Implementation of the RuleLoader.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from enum import Enum
from typing import Any, Callable

import pandas as pd

from dependencies.dp3config import load_attr_spec
from ruleloader.abstractsyntaxtree import (
    ASTBase,
    ASTConstant,
    ASTDict,
    ASTFloat,
    ASTFunction,
    ASTKeyVal,
    ASTList,
    ASTNode,
    ASTProperty,
    ASTString,
)
from ruleloader.classification_taxonomy import ClassificationTaxonomyManager
from ruleloader.conditions import Condition, ConditionFactory, Function
from ruleloader.dp3ifc import DP3Ifc
from ruleloader.exceptions import (
    PrecedenceException,
    PrecedenceMissingRuleException,
    RuleLoaderException,
    SemanticException,
    SyntaxException,
    SyntaxManyException,
)
from ruleloader.rules import AbstractBaseRule, AbstractRuleFactory, OfflineRuleFactory
from ruleloader.tokenizer import Token, TokenDebug, Tokenizer, Tokens

logging.basicConfig(level=logging.INFO)


class RuleLoader:  # pylint: disable=too-many-instance-attributes
    """Parses rules from a config file and loads them as callable `Rule` and `RangeRule` objects."""

    expression_token_types = [
        Tokens.word,
        Tokens.property,
        Tokens.string,
        Tokens.float,
        Tokens.colon,
        Tokens.comma,
        Tokens.eof,
        Tokens.open_brack,
        Tokens.close_brack,
        Tokens.open_square,
        Tokens.close_square,
        Tokens.open_curly,
        Tokens.close_curly,
        Tokens.lt,
        Tokens.gt,
        Tokens.eq,
    ]

    def __init__(
        self,
        rule_factory: AbstractRuleFactory,
        dp3_ifc: DP3Ifc = None,
        class_taxonomy: ClassificationTaxonomyManager = None,
    ) -> None:
        self.tokenizer: Tokenizer
        self.token_iterator: Iterator[Token]
        self.token: Any

        self.precedence_stack: list[Token] = []
        self.ast_stack: list[ASTBase] = []

        ruleloader_dir = os.path.dirname(__file__)
        operator_precedence_table_path = os.path.join(ruleloader_dir, "data", "precedence_tbl.csv")
        self.operator_precedence_table = pd.read_csv(operator_precedence_table_path, sep=",", header=0, index_col=0)
        self.precedence_grammar_rules: dict[tuple, tuple[str, Callable]] = {
            ("Val", "and", "Val"): ("Val", self._construct_bool_node),
            ("Val", "or", "Val"): ("Val", self._construct_bool_node),
            ("Val", "in", "Val"): ("Val", self._construct_in_node),
            ("Val", "lt", "Val"): ("Val", self._construct_comparison_node),
            ("Val", "gt", "Val"): ("Val", self._construct_comparison_node),
            ("Val", "eq", "Val"): ("Val", self._construct_comparison_node),
            ("Val", "comma"): ("Val", self.ast_stack.pop),
            ("property",): ("Val", self._construct_property_access),
            ("string",): ("Val", lambda: self._construct_ast_const(ASTString)),
            ("float",): ("Val", lambda: self._construct_ast_const(ASTFloat)),
            ("word", "EmptyBrack"): ("Val", self._construct_ast_function),
            ("open_brack", "close_brack"): ("EmptyBrack", self._construct_empty_bracks),
            ("word", "Val"): ("Val", self._construct_ast_function),
            ("open_brack", "Val", "close_brack"): ("Val", self._remove_bracks),
            ("Val", "comma", "Val"): ("Val", self._construct_ast_list),
            ("open_curly", "KeyVal", "close_curly"): ("Val", self._construct_ast_dict),
            ("KeyVal", "comma", "KeyVal"): ("KeyVal", self._construct_ast_list),
            ("Val", "colon", "Val"): ("KeyVal", self._construct_ast_key_val),
        }

        self.rule_factory = rule_factory
        self.condition_factory = ConditionFactory()

        self.current_classes: list[Enum] = []
        self.base_belief: float
        self.current_conditions: list[Condition] = []
        self.neg_flag = False
        self.err_flag = False

        self.dp3_ifc = dp3_ifc

        if class_taxonomy is None:
            class_taxonomy = ClassificationTaxonomyManager()
        self.class_taxonomy = class_taxonomy

        self.source = ""
        self.rules: list[AbstractBaseRule] = []

    def _next_token(self) -> None:
        self.token = next(self.token_iterator)

    def _assert_token_type(self, token_type: Tokens) -> None:
        if self.token.type != token_type:
            loc = self.token.debug.get_loc() if self.token.debug is not None else None
            raise SyntaxException(self.token, token_type, loc)

    def parse_file(self, filename: str) -> list[AbstractBaseRule]:
        """Parse a config file and return rules."""
        with open(filename) as infile:
            self.source = infile.read()
        self._reset_parser()
        self._parse_rules()
        return self.rules

    def parse_str(self, source: str) -> list[AbstractBaseRule]:
        """Parse a string containing config and return rules."""
        self.source = source
        self._reset_parser()
        self._parse_rules()
        return self.rules

    def _reset_parser(self):
        self.rules = []
        self.tokenizer = Tokenizer(self.source)
        self.token_iterator = iter(self.tokenizer)
        self.err_flag = False

    def _parse_rules(self) -> None:
        r"""
        Parses the rule configuration file.

        config -> ('blank_lines', 'classification_seq', 'EOF')
            predict:	{'WORD', '\\n', 'WHITESPACE', 'EOF'}

        classification_seq -> ('classification', '\\n', 'blank_lines', 'classification_seq')
            predict:	{'WORD', 'NEG'}
        classification_seq -> ()
            predict:	{'EOF'}
        """
        self._next_token()

        if self.token.type not in (Tokens.word, Tokens.newline, Tokens.whitespace, Tokens.eof):
            loc = self.token.debug.get_loc() if self.token.debug is not None else None
            raise SyntaxManyException(self.token, [Tokens.word, Tokens.newline, Tokens.whitespace, Tokens.eof], loc)

        self._blank_lines()

        last_exception = Exception()
        while self.token.type in (Tokens.word, Tokens.neg):
            try:
                self._classification_syntax()

                if self.token.type == Tokens.eof:
                    break

                self._assert_token_type(Tokens.newline)
                self._next_token()

                self._blank_lines()
            except RuleLoaderException as error:
                last_exception = error
                logging.error(error)
                self.err_flag = True
                self._next_token()
                while self.token.type != Tokens.newline or self.tokenizer.seek(1).type not in [Tokens.neg, Tokens.word]:
                    if self.token.type == Tokens.eof:
                        break
                    self._next_token()
                self._next_token()
                continue

        self._assert_token_type(Tokens.eof)

        if self.err_flag:
            logging.error("There were errors in the configuration.")
            raise last_exception

    def _classification_syntax(self) -> None:
        r"""
        classification -> (opt_neg, opt_whitespace, category_path, (',', opt_whitespace, category_path)*, '\\n',
                           'WHITESPACE', 'rule', 'rules')
            predict:	{'WORD', 'NEG'}
        category_path -> 'WORD', 'PROPERTY'*
        """
        self._optional_neg()
        self._optional_whitespace()

        self._assert_token_type(Tokens.word)
        self.current_classes = self._semantic_rule_class_check(self._get_current_class_path())

        while self.token.type == Tokens.comma:
            self._next_token()
            self._optional_whitespace()

            self._assert_token_type(Tokens.word)
            self.current_classes.extend(self._semantic_rule_class_check(self._get_current_class_path()))

        if self.neg_flag:
            self._semantic_check_single_taxonomy()
            all_classes = set(self.class_taxonomy.get_all_classes_of_group(self.current_classes[0]))
            self.current_classes = list(all_classes.difference(self.current_classes))

        self._assert_token_type(Tokens.newline)
        self._next_token()

        self._assert_token_type(Tokens.whitespace)
        self._next_token()

        self._rule_syntax()
        self._rules_syntax()

    def _optional_neg(self) -> None:
        r"""
        optional_neg -> ('NEG', )
               predict:	{'NEG'}
        optional_neg -> ()
               predict:	{!'NEG'}
        """
        self.neg_flag = False
        if self.token.type == Tokens.neg:
            self.neg_flag = True
            self._next_token()

    def _get_current_class_path(self):
        """Read and return the full current class string."""
        current_class = [self.token.value]
        current_debug = self.token.debug
        self._next_token()

        while self.token.type == Tokens.property:
            current_class.append(self.token.value)
            current_debug += self.token.debug
            self._next_token()
        self.token.debug = current_debug
        return "".join(current_class)

    def _semantic_rule_class_check(self, class_id) -> list[Enum]:
        """Attempt to create a class list basend on give `class_id`, raise SemanticException on fail."""
        try:
            return self.class_taxonomy.create_class_list(class_id)
        except KeyError as error:
            loc = self.token.debug.get_loc() if self.token.debug is not None else None
            raise SemanticException(f"Unknown classification type: {class_id}", loc) from error

    def _rule_syntax(self) -> None:
        """
        rule -> ('FLOAT', 'opt_whitespace','rule_development')

        rule_development -> ('basic_rule',)
            predict:	{':'}
        rule_development -> ('range_rule',)
            predict:	{'-'}
        """
        self._assert_token_type(Tokens.float)
        self.base_belief = float(self.token.value)
        self._semantic_belief_check(self.base_belief)
        self._next_token()

        self._optional_whitespace()

        if self.token.type == Tokens.colon:
            self._basic_rule_syntax()
        elif self.token.type == Tokens.dash:
            self._range_rule_syntax()
        else:
            loc = self.token.debug.get_loc() if self.token.debug is not None else None
            raise SyntaxManyException(self.token, [Tokens.colon, Tokens.dash], loc)

    def _semantic_belief_check(self, belief: float) -> None:
        if not 0 <= belief <= 1:
            loc = self.token.debug.get_loc() if self.token.debug is not None else None
            raise SemanticException(f"The given belief {belief} is out of range <0;1>", loc)

    def _basic_rule_syntax(self) -> None:
        r"""
        basic_rule -> (':', 'EXPR', '\\n')
        """
        self._assert_token_type(Tokens.colon)
        self._next_token()

        cond = self._expr_precedence_syntax()

        self._semantic_check_single_taxonomy()
        all_classes = self.class_taxonomy.get_all_classes_of_group(self.current_classes[0])
        self.rules.append(self.rule_factory.create_rule(self.current_classes, all_classes, self.base_belief, cond))

        self._assert_token_type(Tokens.newline)
        self._next_token()

    def _semantic_check_single_taxonomy(self) -> None:
        """Check that the entered classes are from the same taxonomy."""
        unique_roots = {self.class_taxonomy.get_root_class_of(class_name) for class_name in self.current_classes}
        if len(unique_roots) > 1:
            loc = self.token.debug.get_loc() if self.token.debug is not None else None
            raise SemanticException(f"The class set {self.current_classes} implicitly mixes taxonomies.", loc)

    def _range_rule_syntax(self) -> None:
        r"""
        range_rule -> (
            '-', 'WHITESPACE', 'FLOAT', ':', 'opt_newline',
            'opt_whitespace', '[', 'opt_newline',
            'opt_whitespace', 'expr_list', 'opt_newline',
            'opt_whitespace', ']', '\\n')
        """
        self._assert_token_type(Tokens.dash)
        self._next_token()

        self._assert_token_type(Tokens.whitespace)
        self._next_token()

        self._assert_token_type(Tokens.float)
        max_belief = float(self.token.value)
        self._semantic_belief_check(max_belief)
        self._semantic_max_belief_gt_base_check(max_belief)
        self._next_token()

        self._assert_token_type(Tokens.colon)
        self._next_token()

        self._optional_newline()
        self._optional_whitespace()

        self._assert_token_type(Tokens.open_square)
        self._next_token()

        self._optional_newline()
        self._optional_whitespace()
        self._expr_list_syntax()

        self._semantic_check_single_taxonomy()
        all_classes = self.class_taxonomy.get_all_classes_of_group(self.current_classes[0])
        self.rules.append(
            self.rule_factory.create_range_rule(
                self.current_classes, all_classes, self.base_belief, max_belief, self.current_conditions
            )
        )

        self.current_conditions = []

        self._optional_whitespace()

        self._assert_token_type(Tokens.close_square)
        self._next_token()
        self._assert_token_type(Tokens.newline)
        self._next_token()

    def _semantic_max_belief_gt_base_check(self, max_belief: float) -> None:
        if max_belief <= self.base_belief:
            loc = self.token.debug.get_loc() if self.token.debug is not None else None
            raise SemanticException(
                f"The given upper belief limit {max_belief} " f"must be greater than base belief {self.base_belief}",
                loc,
            )

    def _rules_syntax(self) -> None:
        r"""
        rules -> ('WHITESPACE', 'rule', 'rules')
            predict:	{'WHITESPACE'}
        rules -> ()
            predict:	{'\\n', 'EOF'}
        """
        if self.token.type == Tokens.whitespace:
            self._next_token()

            self._rule_syntax()
            self._rules_syntax()
            return
        if self.token.type in (Tokens.newline, Tokens.eof):
            return
        loc = self.token.debug.get_loc() if self.token.debug is not None else None
        raise SyntaxManyException(self.token, [Tokens.whitespace, Tokens.newline, Tokens.eof], loc)

    def _expr_list_syntax(self) -> None:
        r"""
        expr_list -> ('EXPR', '\\n', 'WHITESPACE', 'expr_list')
            predict:	{'EXPR'}
        expr_list -> ()
            predict:	{'WHITESPACE', ']'}
        """
        if self.token.type in (Tokens.whitespace, Tokens.close_square):
            return
        if self._precedence_accepts():
            cond = self._expr_precedence_syntax()
            self.current_conditions.append(cond)

            self._optional_newline()
            self._optional_whitespace()

            self._expr_list_syntax()
            return
        loc = self.token.debug.get_loc() if self.token.debug is not None else None
        raise SyntaxManyException(self.token, [Tokens.whitespace, Tokens.close_square], loc)

    def _optional_newline(self) -> None:
        r"""
        opt_newline -> ('\\n',)
            predict:	{'\\n'}
        opt_newline -> ()
            predict:	{<anything but \\n>}
        """
        if self.token.type == Tokens.newline:
            self._next_token()
            return

    def _optional_whitespace(self) -> None:
        r"""
        opt_whitespace -> ('WHITESPACE',)
            predict:	{'WHITESPACE'}
        opt_whitespace -> ()
            predict:	{<anything but 'WHITESPACE'>}
        """
        if self.token.type == Tokens.whitespace:
            self._next_token()
            return

    def _blank_lines(self) -> None:
        r"""
        blank_lines -> ('opt_whitespace', '\\n', 'blank_lines')
            predict:	{'\\n', 'WHITESPACE'}
        blank_lines -> ()
            predict:	{'WORD', 'EOF'}
        """
        if self.token.type in (Tokens.newline, Tokens.whitespace):
            self._optional_whitespace()

            self._assert_token_type(Tokens.newline)
            self._next_token()

            self._blank_lines()
            return
        if self.token.type in (Tokens.word, Tokens.eof, Tokens.neg):
            return
        raise SyntaxManyException(
            self.token,
            [Tokens.newline, Tokens.whitespace, Tokens.word, Tokens.eof, Tokens.neg],
            self.token.debug.get_loc() if self.token.debug is not None else None,
        )

    def _precedence_accepts(self) -> bool:
        return not (
            self.token.type in (Tokens.newline, Tokens.close_square)
            or (self.token.type == Tokens.comma and self.tokenizer.seek(1).type == Tokens.newline)
            or (self.token.type == Tokens.whitespace and self.tokenizer.seek(1).type == Tokens.close_square)
        )

    def _expr_precedence_syntax(self) -> Condition:
        """Parse an expression using operator-precedence grammar."""
        # pylint: disable=comparison-with-callable
        self._precedence_init()
        while True:
            logging.debug("precedence_stack=%s", self.precedence_stack)
            top_index, top_term = self._precedence_top()
            logging.debug("top_term=%s, token=%s", top_term, self.token)
            if top_term.annot == Tokens.eof.name and self.token.annot == Tokens.eof.name:
                return self._precedence_finish()
            action = self.operator_precedence_table.loc[top_term.annot, self.token.annot]
            if action == "=":
                self._precdence_shift()
            elif action == "<":
                self._precedence_begin_handle(top_index)
            elif action == ">":
                self._precedence_end_handle()
            else:
                loc = self.token.debug.get_loc() if self.token.debug is not None else None
                raise PrecedenceException(self.token, top_term, loc)

    def _precedence_init(self) -> None:
        """Initialize precedence syntax analysis."""
        self.precedence_stack.clear()
        self.precedence_stack.append(Token(Tokens.eof, "$", Tokens.eof.name))

        self.ast_stack.clear()

        if self.token.type == Tokens.whitespace:
            self._next_token()
        self._precedence_annotate_token()

    def _precedence_top(self) -> tuple[int, Token]:
        """Return the top term."""
        return next(
            (
                (index, token)
                for index, token in zip(range(len(self.precedence_stack), 0, -1), reversed(self.precedence_stack))
                if token.type in self.expression_token_types
            )
        )

    def _precedence_begin_handle(self, top_index: int) -> None:
        self.precedence_stack.insert(top_index, Token(Tokens("<"), "<", "<"))
        self._precedence_push_and_replace_token()

    def _precdence_shift(self) -> None:
        self._precedence_push_and_replace_token()

    def _precedence_end_handle(self) -> None:
        """Perform 'end handle' operation of precedence syntax analysis."""
        handle_content, debug = self._precedence_get_stack_offering()
        logging.debug("handle_content=%s", handle_content)
        logging.debug("ast_stack=%s", self.ast_stack)
        if handle_content not in self.precedence_grammar_rules:
            raise PrecedenceMissingRuleException(handle_content, debug.get_loc() if debug is not None else None)

        nonterm, ast_stack_operation = self.precedence_grammar_rules[handle_content]
        self.precedence_stack.append(Token(nonterm, "", nonterm, debug))

        ast_stack_operation()

    def _precedence_push_and_replace_token(self) -> None:
        self.precedence_stack.append(self.token)
        self.ast_stack.append(self.token)
        self._precedence_next_token()

    def _precedence_next_token(self) -> None:
        """Get next token and annotate it for precedence syntax analysis."""
        while True:
            self._next_token()
            if self.token.type != Tokens.whitespace:
                self._precedence_annotate_token()
                return

    def _precedence_annotate_token(self) -> None:
        """Annotate token for precedence syntax analysis."""
        if self.token.value in ["in", "or", "and"]:
            self.token.annot = self.token.value
        elif self.token.type.name not in self.operator_precedence_table.columns:
            self.token.annot = Tokens.eof.name
        else:
            self.token.annot = self.token.type.name

    def _precedence_get_stack_offering(self) -> tuple[tuple[str, ...], TokenDebug | None]:
        """Get a tuple of items that are above 'begin handle' on precedence stack."""
        stack_offering = []
        debug: TokenDebug | None = None
        while self.precedence_stack:
            token = self.precedence_stack.pop()
            if token.type == Tokens.precedence_beg_handle:
                break
            stack_offering.append(token.annot)
            if debug is None:
                debug = token.debug
            else:
                debug += token.debug
        return tuple(reversed(stack_offering)), debug

    def _precedence_finish(self) -> Condition:
        """Finalize the precedence syntax analysis, create a condition and return it."""
        if len(self.ast_stack) == 0:
            raise RuleLoaderException("A condition must follow the colon.", self.token.debug.get_loc())
        result = self.ast_stack.pop()
        if isinstance(result, ASTFunction) and result.runtime_arg_type is not None:
            raise SemanticException(
                f"{result} is not a standalone function. " f"Did you include all required arguments?",
                result.debug.get_loc(),
            )
        if result.type != bool:
            raise SemanticException(f"A {type(result)} {result} is not a valid condition", result.debug.get_loc())
        condition_result = result.to_condition(self.condition_factory)
        logging.debug("condition %s\n\n", str(condition_result))
        return condition_result

    def _remove_bracks(self) -> None:
        """Remove brackets from AST."""
        self.ast_stack.pop()
        inside_brackets = self.ast_stack.pop()
        if isinstance(inside_brackets, ASTList):
            inside_brackets.complete = True
        self.ast_stack.pop()
        self.ast_stack.append(inside_brackets)

    def _construct_empty_bracks(self) -> None:
        """Construct empty brackets in AST."""
        left = self.ast_stack.pop()
        right = self.ast_stack.pop()
        self.ast_stack.append(ASTList([], left.debug + right.debug))

    def _construct_comparison_node(self) -> None:
        """Construct comparison node from AST."""
        node = self._get_ast_node()
        node.type = bool
        if isinstance(node.left, (ASTDict, ASTList)):
            raise SemanticException("Cannot compare objects of this type", node.left.debug.get_loc())
        if isinstance(node.right, (ASTDict, ASTList)):
            raise SemanticException("Cannot compare objects of this type", node.right.debug.get_loc())
        self._validate_convertable_types(node.left, node.right, check_iterable=True)
        self.ast_stack.append(node)

    def _validate_convertable_types(self, left: ASTBase, right: ASTBase, check_iterable=False) -> None:
        """Validate that left and right node can be converted to same type, raise SemanticException otherwise."""
        if self.dp3_ifc is None:
            return
        left_types = (
            left.get_possible_types() if not isinstance(left, ASTProperty) else left.get_dp3_types(self.dp3_ifc)
        )
        right_types = (
            right.get_possible_types() if not isinstance(right, ASTProperty) else right.get_dp3_types(self.dp3_ifc)
        )
        if not isinstance(left_types, set) or not isinstance(right_types, set):
            raise SemanticException(
                f"Unable to convert types {left_types} and {right_types}.", (left.debug + right.debug).get_loc()
            )

        if check_iterable:
            if isinstance(left, ASTProperty) and self.dp3_ifc.is_iterable(left.attr):
                left_types = {list[a_type] for a_type in left.get_dp3_types(self.dp3_ifc)}  # type: ignore
            if isinstance(right, ASTProperty) and self.dp3_ifc.is_iterable(right.attr):
                right_types = {list[a_type] for a_type in right.get_dp3_types(self.dp3_ifc)}  # type: ignore

        possible = left_types & right_types
        if not possible:
            raise SemanticException(
                f"Unable to convert types {left_types} and {right_types}.", (left.debug + right.debug).get_loc()
            )

    def _construct_bool_node(self) -> None:
        """Construct a bool node from AST."""
        node = self._get_ast_node()
        node.type = bool
        if node.left.type != bool or node.right.type != bool:
            raise SemanticException(
                f"Unable to initialize a boolean operator with {node.left.type} and {node.right.type}.",
                node.debug.get_loc(),
            )
        self.ast_stack.append(node)

    def _construct_in_node(self) -> None:
        """Construct a `in` operator node from AST."""
        node = self._get_ast_node()
        node.type = bool
        if not isinstance(node.right, ASTProperty):
            raise SemanticException("The in operator is only supported for searching properties.", node.debug.get_loc())
        self.ast_stack.append(node)
        if self.dp3_ifc is None:
            return
        if isinstance(node.left, ASTConstant):
            self._validate_convertable_types(node.left, node.right)

        if not self.dp3_ifc.is_iterable(node.right.attr):
            raise SemanticException("The attr given to in operator is not iterable", node.right.debug.get_loc())

        if isinstance(node.left, ASTDict):
            self._validate_dicts_of_types(node.left, node.right)
        if isinstance(node.left, ASTList):
            raise SemanticException(
                "The in operator currently doesn't allow for tuple comparison", node.debug.get_loc()
            )
        if isinstance(node.left, ASTFunction):
            possible = node.right.get_dp3_types(self.dp3_ifc)
            if node.left.runtime_arg_type not in possible:
                raise SemanticException(
                    f"The function {node.left} expects type {node.left.runtime_arg_type}, "
                    f"but would recieve {possible} from {node.right.attr}",
                    node.debug.get_loc(),
                )

    def _validate_dicts_of_types(self, compared: ASTDict, attr: ASTProperty):
        """Validate types of compared dict nodes in AST."""
        if self.dp3_ifc is None:
            return
        compared_types = compared.get_dict_types()
        attr_types: dict[str, type] = attr.get_dp3_types(self.dp3_ifc)  # type: ignore
        for key, type_def in compared_types.items():
            if key not in attr_types:
                raise SemanticException(
                    f"No key '{key}' of attribute '{attr.attr}'.", (compared.debug + attr.debug).get_loc()
                )
            if not {attr_types[key]} & type_def:  # type: ignore
                raise SemanticException(
                    f"No way to convert '{key}' key's type {type_def} to {attr_types[key]},"
                    f" defined by {attr.attr}.",
                    (compared.debug + attr.debug).get_loc(),
                )

    def _get_ast_node(self) -> ASTNode:
        """Return a new AST node from the ast_stack."""
        right = self.ast_stack.pop()
        oper: Token = self.ast_stack.pop()  # type: ignore
        left = self.ast_stack.pop()
        return ASTNode(oper, left, right)

    def _construct_ast_function(self) -> None:
        """Construct a function node from the AST."""
        args = self.ast_stack.pop()
        if not isinstance(args, ASTList):
            args = ASTList([args])
        name: Token = self.ast_stack.pop()  # type: ignore

        if not self.condition_factory.is_valid_function(name.value):
            loc = name.debug.get_loc() if name.debug is not None else None
            raise SemanticException(f"No function called {name.value}.", loc)
        function = ASTFunction(name, args)

        self._validate_args_match_signature(function)
        self.ast_stack.append(function)

    def _validate_args_match_signature(self, function: ASTFunction) -> None:  # pylint: disable=R0912
        """Validate that function arguments fit function signature, SemanticException otherwise."""
        signatures = self.condition_factory.get_function_signatures(function.name.value)
        arg_types = self._get_function_types(function)

        if self.dp3_ifc is not None:
            arg_types_with_iterables = []
            for arg, arg_type in zip(function.args.list, arg_types):
                if isinstance(arg, ASTProperty) and self.dp3_ifc.is_iterable(arg.attr):
                    try:
                        arg_types_with_iterables.append({list[type_var] for type_var in arg_type})  # type: ignore
                    except TypeError as error:
                        raise SemanticException(
                            f"Function args must be of more primitive type than {arg_type}", arg.debug.get_loc()
                        ) from error
                else:
                    arg_types_with_iterables.append(arg_type)  # type: ignore
            arg_types = arg_types_with_iterables  # type: ignore

        for eval_fn, def_args, def_runtime_arg, def_ret_type in signatures:
            # pylint: disable=unnecessary-ellipsis
            if len(def_args) == 2 and def_args[1] is ...:
                for given in arg_types:
                    if def_args[0] not in given:
                        break
                else:
                    return self._signature_success(function, (eval_fn, def_args, def_runtime_arg, def_ret_type))
            if len(arg_types) != len(def_args):
                continue
            for defined, given in zip(def_args, arg_types):
                if defined not in given:
                    break
            else:
                # for loop completed without break, the args match the signature.
                return self._signature_success(function, (eval_fn, def_args, def_runtime_arg, def_ret_type))
        raise SemanticException(
            f"Found no match for signature {function.name.value}{arg_types}",
            (function.name.debug + function.args.debug).get_loc(),
        )

    def _get_function_types(self, function):
        """Return the types of the arguments of the function."""
        arg_types_list = []
        for arg in function.args.list:
            if isinstance(arg, ASTProperty) and self.dp3_ifc is not None:
                arg_types_list.append(arg.get_dp3_types(self.dp3_ifc))
            elif isinstance(arg, ASTDict):
                arg_types_list.append(arg.get_dict_types())
            else:
                arg_types_list.append(arg.get_possible_types())
        return tuple(arg_types_list)

    @staticmethod
    def _signature_success(function: ASTFunction, signature: tuple[Callable, tuple, type | None, type]) -> None:
        """Fit the signature types to the function type annotation."""
        implementation, defined_args, defined_runtime_arg_type, defined_ret_type = signature
        logging.debug(
            "Matched function %s with signature %s",
            function.name.value,
            Function.signature_to_str((defined_args, defined_runtime_arg_type, defined_ret_type)),
        )
        function.runtime_arg_type = defined_runtime_arg_type
        function.type = defined_ret_type
        function.implementation = implementation

    def _construct_property_access(self) -> None:
        """Construct a property access object from the AST."""
        attr = self.ast_stack.pop()
        debug = attr.debug
        attr = attr.value[1:]
        self.ast_stack.append(ASTProperty(attr, debug))

    def _construct_ast_const(self, to_construct) -> None:
        attr = to_construct(self.ast_stack.pop())
        self.ast_stack.append(attr)

    def _construct_ast_list(self) -> None:
        """Construct a ASTList in the AST."""
        second = self.ast_stack.pop()
        self.ast_stack.pop()
        first = self.ast_stack.pop()
        if isinstance(second, (ASTList, ASTDict)):
            raise SemanticException("Nested structures are not allowed.", second.debug.get_loc())
        if isinstance(first, ASTList):
            if first.complete:
                raise SemanticException("Nested structures are not allowed.", first.debug.get_loc())
            first.list.append(second)
            first.debug += second.debug
            self.ast_stack.append(first)
            return
        self.ast_stack.append(ASTList([first, second]))

    def _construct_ast_dict(self) -> None:
        """Construct ASTDict in the AST."""
        self.ast_stack.pop()
        source_list = self.ast_stack.pop()
        self.ast_stack.pop()
        if not isinstance(source_list, ASTList):
            source_list = ASTList([source_list])
        self.ast_stack.append(ASTDict(source_list))

    def _construct_ast_key_val(self) -> None:
        """Construct a Key:Value pair in the AST."""
        second = self.ast_stack.pop()
        self.ast_stack.pop()
        first = self.ast_stack.pop()
        assert isinstance(first, ASTString), "Key must be string"
        self.ast_stack.append(ASTKeyVal(first, second))


if __name__ == "__main__":
    try:
        rf = OfflineRuleFactory()
        ifc = DP3Ifc(load_attr_spec(sys.argv[2])) if len(sys.argv) > 2 else None
        loader = RuleLoader(rule_factory=rf, dp3_ifc=ifc)
        rules = loader.parse_file(sys.argv[1])
        for rule in rules:
            logging.info(rule)
    except RuleLoaderException as err:
        logging.exception(err)
