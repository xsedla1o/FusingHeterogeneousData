"""
Implementation of the classes derived from AbstractBaseRule.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from pyds.pyds import MassFunction
from ruleloader.classification_taxonomy import ClassificationTaxonomyManager
from ruleloader.conditions import Condition
from ruleloader.dp3ifc import DP3Ifc

logging.basicConfig(level=logging.INFO)


def is_generic_class_list(classifications: list[Enum], taxonomy: ClassificationTaxonomyManager):
    """Returns whether the classification is list of 2nd taxonomy level classes."""
    parent_of_all: Enum | None = None
    children = set()
    for classification in classifications:
        children.add(classification)

        parent = taxonomy.get_parent(classification)
        if parent_of_all is None:
            parent_of_all = parent
        elif parent_of_all != parent:
            return False
    if parent_of_all is None:
        return False
    return set(taxonomy.get_children(parent_of_all)) == children


def get_generic_main_class(evaluation: MassFunction, taxonomy: ClassificationTaxonomyManager):
    """Returns main hypothesis transformed to 2nd taxonomy level."""
    for classification_hypothesis, belief in evaluation.items():
        classifications = list(classification_hypothesis)
        return generalize(classifications, taxonomy), belief


def get_main_hypothesis(evaluation: MassFunction, taxonomy: ClassificationTaxonomyManager):
    """Returns main hypothesis of `evaluation`."""
    unpacked = [(list(h), b) for h, b in evaluation.items()]
    all_classes = set(taxonomy.get_all_classes_of_group(unpacked[0][0][0]))
    unpacked = [(h, b) for h, b in unpacked if set(h) != all_classes]
    return max(unpacked, key=lambda x: x[1])


def generalize(cls: list, taxonomy) -> tuple:
    """Returns given `cls` moved one level up the taxonomy."""
    if is_generic_class_list(cls, taxonomy):
        return (taxonomy.get_parent(cls[0]),)
    if set(taxonomy.get_all_classes_of_group(cls[0])) == set(cls):
        return (taxonomy.get_root_class_of(cls[0]),)
    return tuple(cls)


class AbstractBaseRule(ABC):
    """Abstract base class for all rules."""

    __slots__ = "rule_class", "all_classes"

    def __init__(self, rule_class: list[Enum], all_classes):
        self.rule_class = tuple(rule_class)
        self.all_classes = all_classes

    def __call__(self, record: dict, **kwargs) -> MassFunction | None:
        """Evaluate rule for record."""

    @abstractmethod
    def export(self, taxonomy: ClassificationTaxonomyManager) -> str:
        """Returns the rule configuration of the rule."""


class AbstractRule(AbstractBaseRule):
    """Represents an abstract rule, meaning a condition and a classification to assign if condition applies."""

    __slots__ = "maf", "condition", "mass"

    def __init__(self, rule_class: list[Enum], all_classes, mass: float, condition: Condition):
        super().__init__(rule_class, all_classes)
        self.maf = MassFunction({tuple(rule_class): mass, all_classes: 1 - mass})
        self.condition = condition
        self.mass = mass

    def __str__(self) -> str:
        return f"Rule: {self.rule_class}: {self.mass} if\n\t{self.condition}"

    def __repr__(self) -> str:
        return f"Rule(maf={self.maf}, condition={self.condition})"

    def export(self, taxonomy: ClassificationTaxonomyManager) -> str:
        class_list, mass = get_main_hypothesis(self.maf, taxonomy)
        generalized_class_tuple = generalize(class_list, taxonomy)
        return f'{", ".join(str(x) for x in generalized_class_tuple)}\n\t{mass:.3f}: {self.condition}'


class AbstractRangeRule(AbstractBaseRule):  # pylint: disable=too-many-arguments
    """
    Represents an abstract range-rule,
    meaning a set of conditions and a classification to assign if at least one condition applies.
    Assigned belief mass increases with the amount of positive condition results.
    """

    __slots__ = "base_mass", "eased_mass", "conditions"

    def __init__(
        self, rule_class: list[Enum], all_classes, base_mass: float, max_mass: float, conditions: list[Condition]
    ) -> None:
        super().__init__(rule_class, all_classes)
        self.base_mass = base_mass
        self.eased_mass = max_mass - base_mass
        self.conditions = conditions

    def __str__(self) -> str:
        conditions = "\t" + "\n\t".join(str(x) for x in self.conditions)
        return (
            f"RangeRule: {self.rule_class}: {self.base_mass}-{self.base_mass + self.eased_mass} "
            f"based on\n{conditions}"
        )

    def __repr__(self) -> str:
        return (
            f"RangeRule(rule_class={self.rule_class}, base_mass={self.base_mass}, "
            f"eased_mass={self.eased_mass}, conditions={self.conditions})"
        )

    def get_result(self, factors: float):
        """Returns a mass function based on the factors passed."""
        if factors == 0:
            return None

        factor_count = len(self.conditions)
        mass = self.base_mass + self.eased_mass * ease_out_quart(factors / factor_count)
        return MassFunction({self.rule_class: mass, self.all_classes: 1 - mass})

    def export(self, taxonomy: ClassificationTaxonomyManager) -> str:
        conditions = "\n\t\t".join(str(x) for x in self.conditions)
        rule_class, _ = get_generic_main_class(MassFunction({self.rule_class: self.base_mass}), taxonomy)
        return (
            f'{", ".join(str(x) for x in rule_class)}\n'
            f"\t{self.base_mass} - {self.base_mass + self.eased_mass}: [\n"
            f"\t\t{conditions}\n"
            f"\t]"
        )


class AbstractOfflineRule(ABC):
    """Common for all offline rules"""

    __slots__ = ()

    @staticmethod
    def get_attr_value_confidence(record_with_confidence: dict, attribute: str, value) -> float:
        """Returns confidence for given value of given attribute."""
        attr_value_confidence = record_with_confidence[attribute]
        if isinstance(attr_value_confidence, list) and isinstance(value, list):
            confidence_list = [c for val, c in attr_value_confidence if val in value]
            if confidence_list:
                return float(np.prod(confidence_list))
            raise ValueError(f"No such value {value} in {attr_value_confidence}")
        val, c = attr_value_confidence
        if isinstance(value, list) and [x for x in val if x in value] == value:
            return c
        if val != value:
            raise ValueError(f"Unexpected attr value: {repr(val)} (expected {repr(value)})")
        return c

    def get_confidence_coefficient(self, condition: Condition, record: dict, record_with_confidence: dict) -> float:
        """Returns the product of confidence values of all attributes used in the condition."""
        coefficient = 1.0
        triggered = condition.get_trigger_dictionary(record)
        for attr, value in triggered.items():
            coefficient *= self.get_attr_value_confidence(record_with_confidence, attr, value)
            logging.debug(coefficient)
        return coefficient


class AbstractOnlineRule(ABC):
    """Common for all online rules"""

    __slots__ = ()

    @staticmethod
    def get_attr_value_confidence(record: dict, attribute: str, trig_value) -> float:
        """Returns confidence for given value of given attribute."""
        conf = record[f"{attribute}:c"]
        value = record[attribute]
        # Multi-value
        if isinstance(conf, list) and isinstance(trig_value, list):
            confidence_list = [c for val, c in zip(value, conf) if val in trig_value]
            if confidence_list:
                return float(np.prod(confidence_list))
            raise ValueError(f"No such value {trig_value} in {value}")
        # Array
        if isinstance(trig_value, list) and [x for x in value if x in trig_value] == trig_value:
            return conf
        if value != trig_value:
            raise ValueError(f"Unexpected attr value: {repr(value)} (expected {repr(trig_value)})")
        # Any basic type
        return conf

    def get_confidence_coefficient(self, condition: Condition, record: dict, dp3ifc: DP3Ifc) -> float:
        """Returns the product of confidence values of all attributes used in the condition."""
        coefficient = 1.0
        triggered = condition.get_trigger_dictionary(record)
        for attr, value in triggered.items():
            if dp3ifc.has_confidence(attr):
                coefficient *= self.get_attr_value_confidence(record, attr, value)
        return coefficient

    @abstractmethod
    def get_attr_watchset(self) -> set[str]:
        """Return a set of attributes the rule depends on."""


class OnlineRule(AbstractRule, AbstractOnlineRule):
    """Represents a simple rule, meaning a condition and a classification to assign if condition applies.
    This is the 'online' implementation for use within a running DP3 platform."""

    __slots__ = ("dp3",)

    def __init__(self, rule_class: list[Enum], all_classes, mass: float, condition: Condition, dp3ifc: DP3Ifc) -> None:
        super().__init__(rule_class, all_classes, mass, condition)
        self.dp3 = dp3ifc

    def __call__(self, record: dict, **kwargs) -> MassFunction | None:
        if self.condition(record):
            mass = self.mass * self.get_confidence_coefficient(self.condition, record, self.dp3)
            return MassFunction({tuple(self.rule_class): mass, self.all_classes: 1 - mass})
        return None

    def get_attr_watchset(self) -> set[str]:
        return self.condition.get_attr_watchset()


class OnlineRangeRule(AbstractRangeRule, AbstractOnlineRule):
    """
    Represents an abstract range-rule,
    meaning a set of conditions and a classification to assign if at least one condition applies.
    Assigned belief mass increases with the amount of positive condition results.
    This is the 'online' implementation for use within a running DP3 platform.
    """

    __slots__ = ("dp3",)

    def __init__(
        self,
        rule_class: list[Enum],
        all_classes,
        base_mass: float,
        max_mass: float,
        conditions: list[Condition],
        dp3ifc: DP3Ifc,
    ) -> None:
        super().__init__(rule_class, all_classes, base_mass, max_mass, conditions)
        self.dp3 = dp3ifc

    def __call__(self, record: dict, **kwargs) -> MassFunction | None:
        factors = self.get_factor_count(record=record)
        return self.get_result(factors)

    def get_factor_count(self, record: dict) -> float:
        return sum(self.get_confidence_coefficient(cond, record, self.dp3) for cond in self.conditions if cond(record))

    def get_attr_watchset(self) -> set[str]:
        watchset = set()
        for condition in self.conditions:
            watchset |= condition.get_attr_watchset()
        return watchset


class OfflineRule(AbstractRule, AbstractOfflineRule):
    """Represents a simple rule, meaning a condition and a classification to assign if condition applies.
    This is the 'offline' implementation for use in experiments outside a running DP3 platform."""

    __slots__ = ()

    def __call__(self, record: dict, record_with_confidence: dict = None, **kwargs) -> MassFunction | None:
        if self.condition(record):
            if record_with_confidence is None:
                return self.maf

            mass = self.mass * self.get_confidence_coefficient(self.condition, record, record_with_confidence)
            return MassFunction({tuple(self.rule_class): mass, self.all_classes: 1 - mass})
        return None


class OfflineRangeRule(AbstractRangeRule, AbstractOfflineRule):
    """
    Represents an abstract range-rule,
    meaning a set of conditions and a classification to assign if at least one condition applies.
    Assigned belief mass increases with the amount of positive condition results.
    This is the 'offline' implementation for use in experiments outside a running DP3 platform.
    """

    __slots__ = ()

    def __call__(self, record: dict, record_with_confidence: dict = None, **kwargs) -> MassFunction | None:
        factors = self.get_factor_count(record=record, record_with_confidence=record_with_confidence)
        return self.get_result(factors)

    def get_factor_count(self, record: dict, record_with_confidence: dict = None) -> float:
        if record_with_confidence is None:
            return len([x for x in self.conditions if x(record)])
        return sum(
            self.get_confidence_coefficient(cond, record, record_with_confidence)  # type: ignore
            for cond in self.conditions
            if cond(record)
        )


class AbstractRuleFactory(ABC):
    """Abstract base class for creating subclasses of AbstractRule and AbstractRangeRule."""

    @abstractmethod
    def create_rule(self, rule_class: list[Enum], all_classes, mass: float, condition: Condition) -> AbstractRule:
        """Creates instance of simple Rule."""

    @abstractmethod
    def create_range_rule(
        self, rule_class: list[Enum], all_classes, base_mass: float, max_mass: float, conditions: list[Condition]
    ) -> AbstractRangeRule:
        """Creates instance of RangeRule."""


class OnlineRuleFactory(AbstractRuleFactory):
    """Can create 'online' versions of AbstractRule and AbstractRangeRule."""

    def __init__(self, dp3ifc: DP3Ifc):
        self.dp3ifc = dp3ifc

    def create_rule(self, rule_class: list[Enum], all_classes, mass: float, condition: Condition) -> AbstractRule:
        return OnlineRule(rule_class, all_classes, mass, condition, self.dp3ifc)

    def create_range_rule(
        self, rule_class: list[Enum], all_classes, base_mass: float, max_mass: float, conditions: list[Condition]
    ) -> AbstractRangeRule:
        return OnlineRangeRule(rule_class, all_classes, base_mass, max_mass, conditions, self.dp3ifc)


class OfflineRuleFactory(AbstractRuleFactory):
    """Can create 'offline' versions of AbstractRule and AbstractRangeRule."""

    def create_rule(self, rule_class: list[Enum], all_classes, mass: float, condition: Condition) -> AbstractRule:
        return OfflineRule(rule_class, all_classes, mass, condition)

    def create_range_rule(
        self, rule_class: list[Enum], all_classes, base_mass: float, max_mass: float, conditions: list[Condition]
    ) -> AbstractRangeRule:
        return OfflineRangeRule(rule_class, all_classes, base_mass, max_mass, conditions)


def ease_out_quart(arg: float | int) -> float:
    """
    Easing functions define the change of parameter over time.
    See https://easings.net/#easeOutQuart.
    """
    if arg < 0 or arg > 1:
        raise ValueError(f"The value of x: {arg} must be within the bounds <0, 1>")
    return 1 - pow(1 - arg, 4)
