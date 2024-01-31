"""
Unit tests for the RuleLoader implementation.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
# pylint: disable=missing-function-docstring; unit tests are self-documenting
# pylint: disable=too-many-lines
import os
import re
import sys
import unittest
from abc import ABC
from enum import Enum
from typing import Callable, Optional

from pyds.pyds import MassFunction
from ruleloader.classification_taxonomy import (
    ClassificationTaxonomyManager,
    load_classification_taxonomy,
)
from ruleloader.conditions import Function, Re, TupleEquailityCheck
from ruleloader.dp3ifc import DP3Ifc
from ruleloader.exceptions import (
    PrecedenceException,
    PrecedenceMissingRuleException,
    RuleLoaderException,
    SemanticException,
    SyntaxException,
    SyntaxManyException,
    TokenizerException,
)
from ruleloader.ruleloader import RuleLoader
from ruleloader.rules import (
    AbstractRangeRule,
    AbstractRule,
    OfflineRuleFactory,
    ease_out_quart,
)

sys.path.append("..")
from dependencies.dp3config import (  # noqa: E402, pylint: disable=C0413
    load_and_extend_attr_spec,
)

attr_spec = load_and_extend_attr_spec("config/attr_conf_real")

# Loading of test taxonomy
current_dir = os.path.dirname(__file__)
taxonomy_file_path = os.path.join(current_dir, "data", "test_taxonomy.yaml")
raw_taxonomy = load_classification_taxonomy(taxonomy_file_path)

class_taxonomy = ClassificationTaxonomyManager(raw_taxonomy)
Classes = class_taxonomy.enum

rule_factory = OfflineRuleFactory()

# Constants often repeated in tests
windows_string = "OperatingSystem.Windows"
linux_string = "OperatingSystem.Linux"
macos_string = "OperatingSystem.MacOS"
appliance_string = "Device.Appliance"
server_string = "Device.Server"
router_string = "Device.Router"

windows_class = Classes(windows_string)
linux_class = Classes(linux_string)


def get_all_classes(classification):
    if isinstance(classification, Enum):
        classification = (classification,)
    return class_taxonomy.get_all_classes_of_group(classification[0])


def create_rule(rule_class, mass, condition):
    """Return a rule stub from parameters."""
    if isinstance(rule_class, Enum):
        rule_class = (rule_class,)

    all_classes = class_taxonomy.get_all_classes_of_group(rule_class[0])

    maf = MassFunction({rule_class: mass, tuple(all_classes): 1 - mass})
    return lambda r: maf if condition(r) else None


def create_range_rule(rule_class, base_mass, eased_mass, conditions):
    """Return a range rule stub from parameters."""
    if isinstance(rule_class, Enum):
        rule_class = (rule_class,)

    all_classes = class_taxonomy.get_all_classes_of_group(rule_class[0])
    factor_count = len(conditions)

    def range_rule(record):
        factors = len([x for x in conditions if x(record)])
        if factors > 0:
            mass = base_mass + eased_mass * ease_out_quart(factors / factor_count)
            return MassFunction({rule_class: mass, all_classes: 1 - mass})
        return None

    return range_rule


class SyntaxValid(unittest.TestCase):
    """Valid syntax tests."""

    def test_devel_benchmark(self):
        """Source string is the same as used in initial development."""
        source = f"""
# A comment about this file
# A comment about this class
{appliance_string}
    0.6: 'string' in .property
    0.7: {{'key':'value', 'key2':'value2' }} in .x
    0.747: ('suppose a tuple', 'here') in .x
    0.747: ('suppose a tuple', re('With (a )?regex') ,'here') in .x
    # A comment about this regex
    0.7: {{'key':'value', 'key2': re("(Let's|Let us) say I want a regex here") }} in .x
    0.65: contains('substr') in .y
    0.69: re('regex.*') in .y
    # Comparison operators
    0.42: (.x > 3 or .y < -8) and .z == 0.7
    0.84: .single_value_arg == 'Also works with strings!'
    0.666: 's' in .p and 'r' in .o or 'a' in .b
    # A comment about this range rule
    0.4 - 0.7: [
        's' in .p,
        'k' in .l,
        'm' in .o,
        'w' in .q,
    ]
# A comment on the final line
"""
        loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        loader.parse_str(source)
        self.assertEqual(len(loader.rules), 11)

    def test_multiple_classes(self):
        """Can we parse more than one class?"""
        source = f"""
{windows_string}
    0.7: "windows" in .os

{linux_string}
    0.7: 'Linux' in .os
"""
        loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        loader.parse_str(source)
        self.assertEqual(len(loader.rules), 2)

    def test_set_of_classes(self):
        """Can we parse more than one class in one rule?"""
        source = f"""
{macos_string}, {linux_string}, {windows_string}
    0.6: 'computer' in .hardware_type_ua
"""
        loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        loader.parse_str(source)
        self.assertEqual(len(loader.rules), 1)

    def test_float_compatibility(self):
        """Can we handle float values?"""
        source = f"""
{windows_string}
    0.7: 5 in .x
    0.7: greater(5) in .x
    0.7: greater(5, 6, 7) in .x
    0.7: (5, 6, 7) in .x
"""
        loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        loader.parse_str(source)
        self.assertEqual(len(loader.rules), 4)

    def test_more_comments(self):
        """Can we handle comments?"""
        source = f"""
# The server class is detected by the following rules
{server_string}
    0.8: 'Server' in .tags_by_services
    # A comment at the rule level
    0.3 - 0.6: [
        contains_any_of('NETWORKSERVICE', 'MAIL') in .out_flow_tags
        len(.shodan_monitor) > 0 # A comment following a rule
    ]
"""
        loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        loader.parse_str(source)
        self.assertEqual(len(loader.rules), 2)

    def test_readme_snippets(self):
        """Sanity check, do the snippets in README pass?"""
        source = f"""
{appliance_string}
    0.5: .activity_flows < 10
    0.8: .hostname == 'Roomba'
"""
        loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        loader.parse_str(source)
        self.assertEqual(len(loader.rules), 2)


def _test_raises(cm, source: str, exception: type[BaseException], dp3_ifc: DP3Ifc = None, debug: bool = False):
    """Test that given `source` raises given `exception`."""
    loader = RuleLoader(rule_factory=rule_factory, dp3_ifc=dp3_ifc, class_taxonomy=class_taxonomy)
    with cm(exception) as context_manager:
        loader.parse_str(source)
    if debug:
        print(context_manager.exception, end="\n\n")


class TokenizerError(unittest.TestCase):
    """Test we can throw lexical errors."""

    def test_tokenizer_error(self):
        with self.assertRaises(TokenizerException):
            loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
            loader.parse_str(".%~xd")


class SyntaxErrors(unittest.TestCase):
    """Test various cases of syntax errors."""

    def helper_raises(self, source: str, exception: type[BaseException], debug: bool = False):
        _test_raises(self.assertRaises, source, exception, debug=debug)

    def test_wrong_token(self):
        self.helper_raises(
            f"""
{windows_string}:
    pass
""",
            SyntaxException,
        )

    def test_wrong_many(self):
        self.helper_raises(
            """
123:
    pass
""",
            SyntaxManyException,
        )

    def test_wrong_expression_no_operation(self):
        self.helper_raises(
            f"""
{windows_string}
    0.2: <>
""",
            PrecedenceException,
        )

    def test_wrong_expression_no_rule(self):
        self.helper_raises(
            f"""
{windows_string}
    0.2: '' or or
""",
            PrecedenceMissingRuleException,
        )

    def test_operator_chaining(self):
        self.helper_raises(
            f"""
{windows_string}
    0.2: 1 < .x < 2
""",
            PrecedenceException,
        )
        self.helper_raises(
            f"""
{windows_string}
    0.2: 1 in .x > 2
""",
            PrecedenceException,
        )
        self.helper_raises(
            f"""
{windows_string}
    0.2: 's' in 'samba' in .s
""",
            PrecedenceException,
        )

    def test_empty_expression(self):
        self.helper_raises(
            f"""
{windows_string}
    0.2:
""",
            RuleLoaderException,
        )

    def test_invalid_first_token(self):
        self.helper_raises("[", SyntaxManyException)

    def test_rule_colon_dash(self):
        self.helper_raises(f"{windows_string}\n 1 Little Jagermeister", SyntaxManyException)

    def test_rules_wrong_token(self):
        self.helper_raises(f"{windows_string}\n 0.5:  1 > .a\n[", SyntaxManyException)

    def test_expr_list_wrong_token(self):
        self.helper_raises(f"{windows_string}\n 0.5 - 0.6: [\n 1<.a\n,\n", SyntaxManyException)


class SemanticErrors(unittest.TestCase):
    """Test various cases of semantic errors."""

    def helper_raises(self, source: str, exception: type[BaseException] = SemanticException, debug: bool = False):
        _test_raises(self.assertRaises, source, exception, debug=debug)

    def test_belief_out_of_range(self):
        self.helper_raises(
            f"""
{windows_string}
    2: .x > 1
"""
        )
        self.helper_raises(
            f"""
{windows_string}
    -0.5: .x > 1
"""
        )

    def test_easing_rule_interval(self):
        self.helper_raises(
            f"""
{windows_string}
    0.6 - 0.4: [
        .x > 1
    ]
"""
        )

    def test_unknown_class(self):
        self.helper_raises(
            """
RandomClassX
    1: .x > 1
"""
        )

    def test_unknown_operator(self):
        self.helper_raises(
            f"""
{windows_string}
    1: xor('x') in .x
"""
        )

    def test_wrong_argcount(self):
        self.helper_raises(
            f"""
{windows_string}
    1: re('x+', '-x', 2) in .x
"""
        )
        self.helper_raises(
            f"""
{windows_string}
    1: re() in .x
"""
        )

    def test_constant_is_no_condition(self):
        self.helper_raises(
            f"""
{windows_string}
    1: 1.0
"""
        )
        self.helper_raises(
            f"""
{windows_string}
    1: 'a string is also not'
"""
        )

    def test_list_is_no_condition(self):
        self.helper_raises(
            f"""
{windows_string}
    1: 1,'abc', 3
"""
        )

    def test_tuple_is_no_condition(self):
        self.helper_raises(
            f"""
{windows_string}
    1: ('abc', 2.0)
    1: ('abc', re('x'))
"""
        )

    def test_dict_is_no_condition(self):
        self.helper_raises(
            f"""
{windows_string}
    1: {{'a': 'b'}}
"""
        )

    def test_key_val_is_no_condition(self):
        self.helper_raises(
            f"""
{windows_string}
    1: 'a': 'b'
"""
        )

    def test_non_standalone_predicate_as_standalone(self):
        self.helper_raises(
            f"""
{windows_string}
    1: re('lo+l')
"""
        )

    def test_non_bool_predicate_to_bool_op(self):
        self.helper_raises(
            f"""
{windows_string}
    1: len(.software_ua) and 'x' in .x
"""
        )

    def test_nested_structs_not_supported(self):
        self.helper_raises(
            f"""
{windows_string}
    0.2: {{'a': {{'b': 'c'}}}} in .x
"""
        )
        self.helper_raises(
            f"""
{windows_string}
    0.2: (('3', '4'), 1, 2) == .x
"""
        )
        self.helper_raises(
            f"""
{windows_string}
    0.2: (1, 2, {{'b': 'c'}}) == .x
"""
        )
        self.helper_raises(
            f"""
{windows_string}
    0.2: {{'a': ('3', '4')}} == .x
"""
        )

    def test_in_with_constant(self):
        self.helper_raises(
            f"""
{windows_string}
    0.1: 'a' in ('a', 'b', 'c')
"""
        )

    def test_unknown_predicate(self):
        self.helper_raises(f"{windows_string}\n 0.1: unknown() in .a")

    def test_n_arg_predicate_wrong_type(self):
        self.helper_raises(f"{windows_string}\n 0.1: contains_any_of(1, 2, 3) in .a")


class AdditionalExceptions(unittest.TestCase):
    """Increasing coverage of individual components exceptions."""

    def helper_raises(self, source: str, exception: type[BaseException] = RuleLoaderException, debug: bool = False):
        _test_raises(self.assertRaises, source, exception, debug=debug)

    def test_factory_raises(self):
        r = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        self.assertRaises(SemanticException, r.condition_factory.create_condition, "invalid item", None)

    def test_tuple_eq_eval(self):
        t = TupleEquailityCheck(("a", "b"))
        self.assertFalse(t.eval("xd"))

    def test_predicate_must_recieve_list(self):
        with self.assertRaises(SemanticException):
            Re(None, lambda x, y: True)

    def test_classes_has_str(self):
        self.assertEqual(str(windows_class), windows_string)

    def test_rule_class(self):
        r = rule_factory.create_rule((windows_class,), get_all_classes(windows_class), 0.5, lambda x: False)
        str(r)
        self.assertIsNone(r({}))

    def test_range_rule_class(self):
        r = rule_factory.create_range_rule(
            (windows_class,), get_all_classes(windows_class), 0.5, 0.7, [lambda x: False, lambda x: False]
        )
        str(r)
        self.assertIsNone(r({}))

        self.assertEqual(ease_out_quart(0), 0)
        self.assertEqual(ease_out_quart(1), 1)
        self.assertRaises(ValueError, ease_out_quart, 5)

    def test_tokenizer_seek_after_eof(self):
        self.helper_raises(f"{windows_string}\n 1: 1 in .open_ports\n ", SyntaxException)


class SimpleComparisons(unittest.TestCase):
    """Testing functionality of simple comparison operators."""

    conf_source = f"""
{windows_string}
    0.2: .x > 0
    0.5: .y < -10
    0.7: .z == 42.87
"""

    def setUp(self) -> None:
        self.rule_loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        self.rules = self.rule_loader.parse_str(self.conf_source)

    def test_loader_success(self):
        self.assertEqual(len(self.rules), 3)
        for rule in self.rules:
            self.assertTrue(isinstance(rule, AbstractRule))

    def test_zero(self):
        self.single_rule_value_integrity(0, 0.2, lambda r: r["x"] > 0, "x")

    def test_negative_int(self):
        self.single_rule_value_integrity(1, 0.5, lambda r: r["y"] < -10, "y")

    def test_float(self):
        self.single_rule_value_integrity(2, 0.7, lambda r: r["z"] == 42.87, "z")

    def single_rule_value_integrity(self, index, value, lambda_condition, attr):
        test_values = [x + value for x in [0, 0.0001, -0.0001, 50]]
        rule = self.rules[index]
        lambda_rule = create_rule(windows_class, 0.2, lambda_condition)
        for val in test_values:
            record = {attr: val}
            self.assertEqual(rule(record), lambda_rule(record))


class RuleLoaderTestBase(ABC):
    """Base class including various helper functions."""

    conf_source = """"""
    lambda_rules: list[Callable[..., Optional[MassFunction]]]

    def setUp(self):
        self.rule_loader = RuleLoader(rule_factory=rule_factory, class_taxonomy=class_taxonomy)
        self.rules = self.rule_loader.parse_str(self.conf_source)

    def _test_all_rules(self, record):
        for rule, lambda_rule in zip(self.rules, self.lambda_rules):
            self.assertEqual(rule(record), lambda_rule(record), f"Failed with record={record}")  # pylint: disable=E1101

    def _test_single_rule(self, index, record, debug=False, c_record=None):
        if isinstance(self.rules[index], AbstractRule):
            debug_message = (
                f"{self.rules[index].condition} with {record}:\n\t{self.rules[index](record, c_record)}\n"
                f"{self.lambda_rules[index]} with {record}:\n\t{self.lambda_rules[index](record)}"
            )
        else:
            debug_message = (
                f"{self.rules[index].conditions} with {record}:\n\t{self.rules[index](record, c_record)}\n"
                f"{self.lambda_rules[index]} with {record}:\n\t{self.lambda_rules[index](record)}"
            )
        if debug:
            print(debug_message)
        self.assertEqual(  # pylint: disable=E1101
            self.rules[index](record, c_record),
            self.lambda_rules[index](record),
            f"{debug_message}\nFailed with record={record}",
        )

    @staticmethod
    def match_dict_callable(dictionary, record):
        if all(x in record.keys() for x in dictionary.keys()):
            for k, v in dictionary.items():
                if not v(record[k]):
                    return False
            return True
        return False


class LogicalOperators(RuleLoaderTestBase, unittest.TestCase):
    """Test functionality of logical operators."""

    conf_source = f"""
{windows_string}
    0.2: .x > 0 or .y < 0.6
    0.5: .a == 'str' and (.b == 1 or .c == 1)
"""

    def setUp(self):
        super().setUp()
        self.lambda_rules = [
            create_rule(windows_class, 0.2, lambda r: r["x"] > 0 or r["y"] < 0.6),
            create_rule(windows_class, 0.5, lambda r: r["a"] == "str" and (r["b"] == 1 or r["c"] == 1)),
        ]

    def test_loader_success(self):
        self.assertEqual(len(self.rules), len(self.lambda_rules))
        for rule in self.rules:
            self.assertTrue(isinstance(rule, AbstractRule))

    def test_basic_or(self):
        record = {"x": 1, "y": 0}
        self._test_single_rule(0, record)

        record.update({"x": 0, "y": 0})
        self._test_single_rule(0, record)

        record.update({"x": 0, "y": 1})
        self._test_single_rule(0, record)

    def test_and_or_with_brackets(self):
        record = {"a": "not", "b": 1, "c": 0}
        self._test_single_rule(1, record)

        record.update({"a": "not", "b": 1, "c": 1})
        self._test_single_rule(1, record)

        record.update({"a": "str", "b": 0, "c": 1})
        self._test_single_rule(1, record)


class InOperator(RuleLoaderTestBase, unittest.TestCase):
    """Test functionality of the in operator."""

    conf_source = f"""
{windows_string}
    0.5: 's' in .p
    0.5: {{'k1':'v1'}} in .x
    0.5: {{'k1':'v1', 'k2':'v2' }} in .x
    0.5: ('A', 'B') in .y
"""

    def setUp(self) -> None:
        super().setUp()
        d1 = {"k1": lambda x: x == "v1"}
        d2 = {"k1": lambda x: x == "v1", "k2": lambda x: x == "v2"}
        self.lambda_rules = [
            create_rule(windows_class, 0.5, lambda r: "s" in r["p"]),
            create_rule(windows_class, 0.5, lambda r: any(self.match_dict_callable(d1, x) for x in r["x"])),
            create_rule(windows_class, 0.5, lambda r: any(self.match_dict_callable(d2, x) for x in r["x"])),
            create_rule(windows_class, 0.5, lambda r: ("A", "B") in r["y"]),
        ]

    def test_loader_success(self):
        self.assertEqual(len(self.rules), len(self.lambda_rules))
        for rule in self.rules:
            self.assertTrue(isinstance(rule, AbstractRule))

    def test_empty(self):
        record = {"p": [], "x": [], "y": []}
        self._test_all_rules(record)

    def test_basic_in(self):
        record = {"p": ["a", "b", "c", "s"]}
        self._test_single_rule(0, record)

        record.update({"p": ["a", "b", "c"]})
        self._test_single_rule(0, record)

    def test_dict_in_single(self):
        record = {"x": [{"some": "other", "mostly": "random"}, {"dict": "values"}]}
        self._test_single_rule(1, record)

        record["x"].append({"k1": "Nope"})
        self._test_single_rule(1, record)

        record["x"].append({"k1": "v1", "k2": "v2"})
        self._test_single_rule(1, record)

    def test_dict_in_multiple(self):
        record = {"x": [{"some": "other", "mostly": "random"}, {"dict": "values"}]}
        self._test_single_rule(2, record)

        record["x"].append({"k1": "v1", "k2": "v2"})
        self._test_single_rule(2, record)

    def test_tuple_in(self):
        record = {"y": [("A", "A"), ("B", 3)]}
        self._test_single_rule(3, record)

        record.update({"y": [("A", "B"), ("A", 3)]})
        self._test_single_rule(3, record)


class RangeRuleTest(RuleLoaderTestBase, unittest.TestCase):
    """Test proper functionality of RangeRule."""

    conf_source = f"""
{windows_string}
    0.25 - 0.5: [
        's' in .p,
        'k' in .l,
        'm' in .o,
        'w' in .q,
    ]
"""

    def setUp(self):
        super().setUp()
        self.lambda_rules = [
            create_range_rule(
                windows_class,
                0.25,
                0.25,
                [
                    lambda r: "s" in r["p"],
                    lambda r: "k" in r["l"],
                    lambda r: "m" in r["o"],
                    lambda r: "w" in r["q"],
                ],
            )
        ]

    def test_loader_success(self):
        self.assertEqual(len(self.rules), len(self.lambda_rules))
        for rule in self.rules:
            self.assertTrue(isinstance(rule, AbstractRangeRule))

    def test_false(self):
        record = {"p": [], "l": [], "o": [], "q": []}
        self._test_single_rule(0, record)

    def test_base(self):
        record = {"p": ["s"], "l": [], "o": [], "q": []}
        self._test_single_rule(0, record)

    def test_middle(self):
        record = {"p": ["s"], "l": [], "o": ["l"], "q": []}
        self._test_single_rule(0, record)

    def test_full(self):
        record = {"p": ["s"], "l": ["k"], "o": ["l"], "q": ["w"]}
        self._test_single_rule(0, record)


class Functions(RuleLoaderTestBase, unittest.TestCase):
    """Test individual functions' functionality."""

    conf_source = f"""
{windows_string}
    0.5: re('a+b?') in .y
    0.5: contains('sub') in .y
    0.5: ('a', re('a+')) in .x
    0.5: {{'k1':'v', 'k2': re("Y|N") }} in .z
    0.5: re('a+b?', .a)
    0.5: contains('sub', .a)
    0.5: len(.b) == 2
    0.5: contains_any_of('a', 'b') in .c
"""
    d = {"k1": lambda x: x == "v", "k2": lambda x: re.search("Y|N", x)}

    def setUp(self):
        super().setUp()
        self.lambda_rules = [
            create_rule(windows_class, 0.5, lambda r: any(re.search("a+b?", x) is not None for x in r["y"])),
            create_rule(windows_class, 0.5, lambda r: any("sub" in x for x in r["y"])),
            create_rule(
                windows_class, 0.5, lambda r: any(first == "a" and re.search("a+", second) for first, second in r["x"])
            ),
            create_rule(windows_class, 0.5, lambda r: any(self.match_dict_callable(self.d, x) for x in r["z"])),
            create_rule(windows_class, 0.5, lambda r: re.search("a+b?", r["a"])),
            create_rule(windows_class, 0.5, lambda r: "sub" in r["a"]),
            create_rule(windows_class, 0.5, lambda r: len(r["b"]) == 2),
            create_rule(windows_class, 0.5, lambda r: any(s in string for s in ("a", "b") for string in r["c"])),
        ]

    def test_loader_success(self):
        self.assertEqual(len(self.rules), len(self.lambda_rules))
        for rule in self.rules:
            self.assertTrue(isinstance(rule, AbstractRule))

    def test_re_predicate(self):
        self._test_single_rule(0, {"y": []})
        self._test_single_rule(0, {"y": ["b", "ba", "bb"]})

        self._test_single_rule(0, {"y": ["aaab"]})
        self._test_single_rule(0, {"y": ["a"]})
        self._test_single_rule(0, {"y": ["aa"]})

    def test_re_predicate_standalone(self):
        self._test_single_rule(4, {"a": ""})
        self._test_single_rule(4, {"a": "baba"})

        self._test_single_rule(4, {"a": "a"})
        self._test_single_rule(4, {"a": "aaaaab"})

    def test_contains_predicate(self):
        self._test_single_rule(1, {"y": []})
        self._test_single_rule(1, {"y": ["a", "b", "c"]})

        self._test_single_rule(1, {"y": ["sub"]})
        self._test_single_rule(1, {"y": ["a substring"]})
        self._test_single_rule(1, {"y": ["a", "b", "a substring", "c"]})

    def test_contains_predicate_standalone(self):
        self._test_single_rule(5, {"a": ""})
        self._test_single_rule(5, {"a": "nada"})

        self._test_single_rule(5, {"a": "sub"})
        self._test_single_rule(5, {"a": "inside a [sub]string"})

    def test_predicate_in_tuple(self):
        """0.5: ('a', re('a+')) in .x"""
        self._test_single_rule(2, {"x": []})
        self._test_single_rule(2, {"x": [("one", "two"), ("three", "four")]})

        self._test_single_rule(2, {"x": [("a", "aaaa")]})
        self._test_single_rule(2, {"x": [("a", "aaaaaaaaaaaaaa")]})
        self._test_single_rule(2, {"x": [("a", "b"), ("a", "aaaaaaaaaaaaaa"), ("c", "d")]})

    def test_predicate_in_dict(self):
        """0.5: {'k1':'v', 'k2': re("Y|N") } in .z"""
        self._test_single_rule(3, {"z": []})
        self._test_single_rule(3, {"z": [{"k3": "val"}, {"k1": "v", "k2": "nope"}]})

        self._test_single_rule(3, {"z": [{"k1": "v", "k2": "Y"}]})
        self._test_single_rule(3, {"z": [{"k3": "val"}, {"k1": "v", "k2": "N"}, {"k1": "val"}]})

    def test_len_predicate(self):
        """0.5: len(.b) == 2"""
        self._test_single_rule(6, {"b": []})
        self._test_single_rule(6, {"b": [1, 2, 3]})

        self._test_single_rule(6, {"b": ["just", "right"]})

    def test_contains_any_of_predicate(self):
        """0.5: contains_any_of('a', 'b') in .c"""
        self._test_single_rule(7, {"c": []})
        self._test_single_rule(7, {"c": ["sds", "cdc", "xd"]})

        self._test_single_rule(7, {"c": ["jast", "raijt"]})
        self._test_single_rule(7, {"c": ["jbt", "rbt"]})


class NumberFunctionTest(Function):
    """Mock function having numerical inputs."""

    __slots__ = ()
    name = "greater"

    def __init__(self, args, _):
        super().__init__(args, None)
        self.args = args[0]

    def eval(self, record):
        return record > self.args(record)

    @classmethod
    def get_signature(cls):
        return [
            (None, (int,), int, bool),
            (None, (float,), float, bool),
            (None, (float, float, float), float, bool),
        ]

    def get_attr_watchset(self):
        return set()


class NumberFunctions(RuleLoaderTestBase, unittest.TestCase):
    """Test functionality of functions with numerical inputs."""

    conf_source = f"""
{windows_string}
    0.5: greater(1) in .y
    0.5: ('a', greater(2.0)) in .x
    0.5: {{'k': greater(-0.1)}} in .z
"""
    d = {"k": lambda x: x > -0.1}

    def setUp(self):
        super().setUp()
        self.lambda_rules = [
            create_rule(windows_class, 0.5, lambda r: any(x > 1 for x in r["y"])),
            create_rule(windows_class, 0.5, lambda r: any(first == "a" and second > 2 for first, second in r["x"])),
            create_rule(windows_class, 0.5, lambda r: any(self.match_dict_callable(self.d, x) for x in r["z"])),
        ]

    def test_basic_case(self):
        self._test_single_rule(0, {"y": []})
        self._test_single_rule(0, {"y": [0, 1, -10]})

        self._test_single_rule(0, {"y": [0, 5, -10]})

    def test_inside_tuple(self):
        self._test_single_rule(1, {"x": []})
        self._test_single_rule(1, {"x": [("c", 0), ("a", 0), ("b", 5)]})

        self._test_single_rule(1, {"x": [("c", 0), ("a", 5), ("b", 0)]})

    def test_inside_dict(self):
        self._test_single_rule(2, {"z": []})
        self._test_single_rule(2, {"z": [{"k": -10, "l": -10, "m": 0}]})

        self._test_single_rule(2, {"z": [{"k": 0, "l": -10, "m": 0}]})


class DP3Types(unittest.TestCase):
    """Test correct checking of types against the DP3 configuration."""

    ifc = DP3Ifc(attr_spec)

    def helper_raises(self, source: str, exception: type[BaseException] = SemanticException, debug: bool = False):
        _test_raises(self.assertRaises, source, exception, self.ifc, debug)

    def test_nonexistent_attr(self):
        self.helper_raises(
            f"""
{windows_string}
    0.5: 1 in .y
"""
        )

    def test_wrong_type_list(self):
        self.helper_raises(
            f"""
{windows_string}
    0.5: 'string' in .open_ports
"""
        )

    def test_wrong_type_dict(self):
        self.helper_raises(
            f"""
{windows_string}
    0.5: {{"name": 2.0, "value":""}} in .sdp_label
"""
        )

    def test_wrong_key_dict(self):
        self.helper_raises(
            f"""
{windows_string}
    0.5: {{"port": 2.0, "value":""}} in .dnssd_service
"""
        )

    def test_correct_type_list(self):
        source = f"""
{windows_string}
    0.5: 22 in .open_ports
"""
        r = RuleLoader(rule_factory=rule_factory, dp3_ifc=self.ifc, class_taxonomy=class_taxonomy)
        r.parse_str(source)
        self.assertEqual(len(r.rules), 1)

    def test_correct_type_dict(self):
        source = f"""
{windows_string}
    0.5: {{"port": 22, "service": "ssh"}} in .dnssd_service
"""
        r = RuleLoader(rule_factory=rule_factory, dp3_ifc=self.ifc, class_taxonomy=class_taxonomy)
        r.parse_str(source)
        self.assertEqual(len(r.rules), 1)

    def test_wrong_comparison(self):
        self.helper_raises(
            f"""
{windows_string}
    0.5: 'string' > .open_ports
"""
        )

    def test_wrong_bool_op(self):
        self.helper_raises(
            f"""
{windows_string}
    0.5: 'string' and .open_ports
"""
        )

    def test_predicate(self):
        self.helper_raises(
            f"""
{windows_string}
    0.5: re('string') in .open_ports
"""
        )
        self.helper_raises(
            f"""
{windows_string}
    0.5: re('pattern', .software_name_ua)
"""
        )

    def test_tuple_in(self):
        self.helper_raises(f"{windows_string}\n 1: ('a', 'b') in .open_ports")

    def test_nonmatching_types(self):
        self.helper_raises(f"{windows_string}\n 1: {{'Hello':'there'}} == .hostname")
        self.helper_raises(f"{windows_string}\n 1: .hostname == {{'Hello':'there'}}")
        self.helper_raises(f"{windows_string}\n 1: 'Hello there' in .dnssd_service")


class ConfidencePropagationTests(RuleLoaderTestBase, unittest.TestCase):
    """Test the confidence propagation functionality."""

    conf_source = f"""
{windows_string}
    1: 'a' == .a
    1: 'a' == .a and 'b' == .b
    1: 'a' in .a and 'b' in .a
"""

    def setUp(self):
        super().setUp()
        self.lambda_rules = [
            create_rule(windows_class, 0.5, lambda r: "a" == r["a"]),
            create_rule(windows_class, 0.25, lambda r: "a" == r["a"] and "b" == r["b"]),
            create_rule(windows_class, 0.25, lambda r: "a" in r["a"] and "b" in r["a"]),
        ]

    def test_basic_conf(self):
        self._test_single_rule(0, {"a": "a"}, False, {"a": ("a", 0.5)})

    def test_logical_merge(self):
        self._test_single_rule(1, {"a": "a", "b": "b"}, False, {"a": ("a", 0.5), "b": ("b", 0.5)})

    def test_multivalue_merge(self):
        self._test_single_rule(2, {"a": ["a", "b"]}, False, {"a": [("a", 0.5), ("b", 0.5)]})


class TaxonomyEnhancementsTests(unittest.TestCase):
    """Test features that were added later, in `taxonomy_enhancements` branch."""

    def helper_raises(self, source: str, exception: type[BaseException] = SemanticException, debug: bool = False):
        _test_raises(self.assertRaises, source, exception, debug=debug)

    def test_combined_taxonomies(self):
        self.helper_raises(
            f"""
{windows_string}, {router_string}
    0.5: 'a' == .a
""",
            SemanticException,
        )


if __name__ == "__main__":
    unittest.main()
