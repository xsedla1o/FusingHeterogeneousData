"""
A set of functions implementing dataset classification.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
# Disabling complicated code suggestions - will not change classify_device_profile().
# pylint: disable=R0912,R0913,R0914,R0915

from collections import defaultdict
from datetime import timedelta
from enum import Enum
from functools import partial
from typing import Optional

import numba
import numpy as np
import pandas as pd

from dependencies.profile_history_management import (
    get_attr_spec,
    get_conf_at_t_fast,
    get_profile_first_active_time,
    get_profile_last_active_time,
    get_profile_values_without_and_with_confidence,
    get_profile_values_without_and_with_confidence_preloaded,
    get_value_count,
)
from pyds.pyds import MassFunction
from pyds_utilities import (
    are_conflicting_classifications,
    get_main_hypothesis,
    get_majority_hypothesis,
    reduce,
    unite_evaluations,
    unpack_hypothesis_belief_pair,
)
from ruleloader.classification_taxonomy import ClassificationTaxonomyManager
from ruleloader.rules import AbstractBaseRule


def classify_device_profile(  # noqa: C901
    profile, rules: list[AbstractBaseRule], taxonomy: ClassificationTaxonomyManager
):
    """
    Returns a classification profile for given device profile.

    This is an optimized and streamlined version of Classifier.classify_device().
    """
    interval_delta = np.timedelta64(10, "m")

    min_time = floor_time_minutes(np.datetime64(get_profile_first_active_time(profile)))
    max_time = floor_time_minutes(np.datetime64(get_profile_last_active_time(profile))) + interval_delta

    # Contains the considered times as np.datetime64[m].
    time_arr = np.arange(min_time, max_time, interval_delta)

    # For storing values and confidences at all considered times.
    profile_arr = np.array([{} for _ in range(time_arr.size)])

    # For storing values and confidences at all considered times, but considering only strict t1 and t2 validity.
    strict_time_profile_arr = np.array([{} for _ in range(time_arr.size)])

    for attribute, value in profile.items():
        # If attribute has no history, add it to all constructed profiles.
        sel_attr_spec = get_attr_spec(attribute)
        if not sel_attr_spec.history:
            for i in range(profile_arr.size):
                profile_arr[i][attribute] = value
                strict_time_profile_arr[i][attribute] = value
            continue

        pre_validity = np.timedelta64(2, "h")
        post_validity = np.timedelta64(4, "h")
        strict_validity = np.timedelta64(0, "h")

        if sel_attr_spec.multi_value:
            datapoint_confidence_getter = get_datapoint_confidences_multi_value
        else:
            datapoint_confidence_getter = get_datapoint_confidences

        for datapoint_item in value:
            # Unpack datapoint.
            t1, t2, c, v = tuple(datapoint_item)
            t1 = np.datetime64(t1)
            t2 = np.datetime64(t2)

            # Floor datapoint intervals.
            floored_start_t = floor_time_minutes(t1 - pre_validity)
            floored_end_t = floor_time_minutes(t2 + post_validity)

            floored_t1 = floor_time_minutes(t1)
            floored_t2 = floor_time_minutes(t2)

            # Add value with confidence to existing profile database.
            datapoint_confidence_getter(
                attribute,
                t1,
                t2,
                c,
                v,
                floored_start_t,
                floored_end_t,
                min_time,
                time_arr,
                interval_delta,
                profile_arr,
                pre_validity,
                post_validity,
            )

            # Do the same, but without any pre- and post-validity.
            datapoint_confidence_getter(
                attribute,
                t1,
                t2,
                c,
                v,
                floored_t1,
                floored_t2,
                min_time,
                time_arr,
                interval_delta,
                strict_time_profile_arr,
                strict_validity,
                strict_validity,
            )

    device_eval_profile: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for i in range(profile_arr.size):
        without_conf, with_conf = get_profile_values_without_and_with_confidence_preloaded(profile_arr[i])
        t1 = pd.Timestamp(time_arr[i])
        t2 = t1 + timedelta(minutes=10)

        # Evaluate the current profile.
        profile_evaluations = []
        for rule in rules:
            try:
                evaluation = rule(without_conf, record_with_confidence=with_conf)
            except (TypeError, ValueError) as e:
                raise Exception(
                    "In rule", rule, "\nConsider looking at the rule and the configuration of attributes used."
                ) from e
            if evaluation is not None:
                profile_evaluations.append(evaluation)

        if not profile_evaluations:
            continue

        # Split evaluations by taxonomy root into filtered lists.
        filtered_evaluations = defaultdict(list)
        for evaluation in profile_evaluations:
            main_hypothesis = tuple(get_main_hypothesis(evaluation))[0]
            root = taxonomy.get_root_class_of(main_hypothesis)
            filtered_evaluations[root].append(evaluation)

        # Append evaluations and additional required metrics to the overall device profile.
        for taxonomy_root, evaluation_list in filtered_evaluations.items():
            target_profile = device_eval_profile[taxonomy_root]

            target_profile["rule_evaluations"].append([t1, t2, 1, evaluation_list])

            united_evaluation = unite_evaluations(evaluation_list).pignistic()
            taxonomy_evaluation = get_taxonomy_from_evaluation(taxonomy, united_evaluation)
            target_profile["taxonomy_evaluation"].append([t1, t2, 1, taxonomy_evaluation])

            conflicting_evaluations = are_conflicting_classifications(evaluation_list)
            target_profile["conflicting_evaluation"].append([t1, t2, 1, conflicting_evaluations])

            # Count active attributes.
            active_attribute_count = 0
            profile_values, _ = get_profile_values_without_and_with_confidence_preloaded(strict_time_profile_arr[i])
            for value in profile_values.values():
                if isinstance(value, list):
                    active_attribute_count += 1 if value else 0
                else:
                    active_attribute_count += 1 if value is not None else 0
            target_profile["active_attributes"].append([t1, t2, 1, active_attribute_count])

    # Transform the list values to np.arrays.
    for taxonomy_root, subprofile in device_eval_profile.items():
        subprofile.update({k: np.array(v, dtype=object) for k, v in subprofile.items()})

    return device_eval_profile


def get_datapoint_confidences(
    attribute, t1, t2, c, v, from_t, to_t, min_t, time_arr, interval_delta, target_arr, pre, post
):
    """
    Gets datapoint confidences at specified times.
    Saves them as tuples of (v, confidence) into target_arr.
    Use for single-value attributes.

    :type attribute: str
    :type t1: np.datetime64
    :type t2: np.datetime64
    :type c: float
    :type v: Any
    :type from_t: np.datetime64
    :type to_t: np.datetime64
    :type min_t: np.datetime64
    :type time_arr: np.ndarray
    :type interval_delta: np.timedelta64
    :type target_arr: np.ndarray
    :type pre: np.timedelta64
    :type post np.timedelta64
    """
    for i in get_time_indexes(from_t, to_t, min_t, time_arr.size, interval_delta):
        conf = get_conf_at_t_fast(c, time_arr[i], t1, t2, pre_val=pre, post_val=post)
        if conf == 0:
            continue
        if attribute not in target_arr[i] or target_arr[i][attribute][1] < conf:
            target_arr[i][attribute] = v, conf


def get_datapoint_confidences_multi_value(
    attribute, t1, t2, c, v, from_t, to_t, min_t, time_arr, interval_delta, target_arr, pre, post
):
    """
    Gets datapoint confidences at specified times.
    Saves them as tuples of (v, confidence) into target_arr.
    Use for multi-value attributes.

    :type attribute: str
    :type t1: np.datetime64
    :type t2: np.datetime64
    :type c: float
    :type v: Any
    :type from_t: np.datetime64
    :type to_t: np.datetime64
    :type min_t: np.datetime64
    :type time_arr: np.ndarray
    :type interval_delta: np.timedelta64
    :type target_arr: np.ndarray
    :type pre: np.timedelta64
    :type post np.timedelta64
    """
    for i in get_time_indexes(from_t, to_t, min_t, time_arr.size, interval_delta):
        conf = get_conf_at_t_fast(c, time_arr[i], t1, t2, pre_val=pre, post_val=post)
        if conf == 0:
            continue
        if attribute not in target_arr[i]:
            target_arr[i][attribute] = [(v, conf)]
        else:
            for j, uniq in enumerate(target_arr[i][attribute]):
                v_, c_ = uniq
                if v_ == v:
                    if c_ <= conf:
                        target_arr[i][attribute][j] = v, conf
                    break
            else:
                target_arr[i][attribute].append((v, conf))


def floor_time_minutes(time: np.datetime64, interval_in_minutes=10) -> np.datetime64:
    """Floor the given time value to a given interval in minutes."""
    time_as_delta = time.astype("timedelta64[us]")
    m_to_us = 60_000_000
    flooring_interval = np.timedelta64(interval_in_minutes * m_to_us, "us")
    floored = (time_as_delta - time_as_delta % flooring_interval).astype("datetime64[us]")
    return floored


@numba.njit
def get_time_indexes(start_time, end_time, target_start_time, target_len, delta):
    """Get the indices to data arrays with given parameters."""
    begin_index = max(0, numba.int64((start_time - target_start_time) / delta))
    end_index = min(target_len, numba.int64((end_time - target_start_time) / delta) + 1)
    return np.arange(begin_index, end_index)


class Classifier:
    """Class responsible for device profile classification."""

    def __init__(self, classification_rules, taxonomy: ClassificationTaxonomyManager):
        self.rules = classification_rules
        self.taxonomy = taxonomy

    def classify_device(self, device_profile: dict, interval_len: int = 10):
        """Classify the entire device_profile history in intervals `interval_len` long."""
        # Get a time range where it makes sense to do classifications.
        min_time = get_profile_first_active_time(device_profile)
        max_time = get_profile_last_active_time(device_profile)

        # Iterate from the min_time to max_time and get evaluation profiles.
        time_increment = timedelta(minutes=interval_len)
        num_intervals = ((max_time - min_time).seconds // 60 // interval_len) + 1
        device_eval_profile: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for t in (min_time + time_increment * x for x in range(num_intervals)):
            self.extend_evaluation_profile_by_interval(device_profile, device_eval_profile, t, t + time_increment)

        # Transform the list values to np.arrays.
        for subprofile in device_eval_profile.values():
            subprofile.update({k: np.array(v, dtype=object) for k, v in subprofile.items()})

        return device_eval_profile

    def extend_evaluation_profile_by_interval(self, device_profile: dict, eval_profile: dict, t1, t2) -> None:
        """Extend existing eval_profile by t2-t1 interval classifying device_profile at time t1."""
        evaluation_list = self.get_profile_evaluation_list(device_profile, t1)
        if not evaluation_list:
            return

        filtered_evaluations = defaultdict(list)
        for evaluation in evaluation_list:
            main_hypothesis = tuple(get_main_hypothesis(evaluation))[0]
            root = self.taxonomy.get_root_class_of(main_hypothesis)
            filtered_evaluations[root].append(evaluation)

        for taxonomy_root, evaluation_list in filtered_evaluations.items():
            target_profile = eval_profile[taxonomy_root]

            target_profile["rule_evaluations"].append([t1, t2, 1, evaluation_list])

            united_evaluation = unite_evaluations(evaluation_list).pignistic()
            taxonomy_evaluation = get_taxonomy_from_evaluation(self.taxonomy, united_evaluation)
            target_profile["taxonomy_evaluation"].append([t1, t2, 1, taxonomy_evaluation])

            conflicting_evaluations = are_conflicting_classifications(evaluation_list)
            target_profile["conflicting_evaluation"].append([t1, t2, 1, conflicting_evaluations])

            active_attribute_count = get_value_count(device_profile, t1, 0, 0)
            target_profile["active_attributes"].append([t1, t2, 1, active_attribute_count])

    def get_profile_evaluation_list(self, device_profile: dict, time) -> list[MassFunction]:
        """Get a list of all evaluations for given profile and time."""
        values_at_time, values_at_time_with_confidence = get_profile_values_without_and_with_confidence(
            device_profile, time
        )
        return self.get_evaluation_list(values_at_time, values_at_time_with_confidence)

    def get_evaluation_list(self, record, record_with_confidence=None, debug=False) -> list[MassFunction]:
        """Get a list of all evaluations for given record."""
        evaluations = []

        for rule in self.rules:
            try:
                evaluation = rule(record, record_with_confidence)
            except TypeError as e:
                raise TypeError("In rule", rule) from e
            if evaluation is not None:
                if debug:
                    print("Triggered", rule)
                evaluations.append(evaluation)

        return evaluations


def get_taxonomy_from_evaluation(
    taxonomy: ClassificationTaxonomyManager, evaluation: MassFunction
) -> list[tuple[Optional[Enum], float]]:
    """
    Returns a taxonomic classification of the provided classification MassFunction.

    :param taxonomy Taxonomy of the classification.
    :param evaluation A normalized MassFunction with single-element hypotheses.
        Use the 'pignistic()' member function to achieve this.
    """
    taxonomy_list: list[tuple[Optional[Enum], float, MassFunction]] = []

    # First pass, going from the leaf level of the taxonomy to the root.
    while len(evaluation) > 1:
        majority = get_majority_hypothesis(evaluation)
        if majority is None:
            taxonomy_list.append((None, 0, evaluation))
        else:
            taxonomy_list.append((*unpack_hypothesis_belief_pair(majority), evaluation))
        evaluation = merge_taxonomy_classes(taxonomy, evaluation)

    majority = get_majority_hypothesis(evaluation)
    taxonomy_list.append((*unpack_hypothesis_belief_pair(majority), evaluation))

    # Second pass, examining the classified tree top-down, filling in None values.
    previous_class = None
    taxonomy_list.reverse()

    def is_child_of_previous(class_frozenset, _, previous) -> bool:
        return taxonomy.get_parent(list(class_frozenset)[0]) == previous

    for i, evaluated in enumerate(taxonomy_list):
        classification, _, evaluation = evaluated

        if classification is not None:  # We already have a classification at this level
            previous_class = classification
            continue

        # We attempt to get majority by reducing the MassFunction to a subset of previous classification's children.
        original_evaluation = evaluation
        evaluation = reduce(evaluation, partial(is_child_of_previous, previous=previous_class))
        majority = get_majority_hypothesis(evaluation)
        if majority is None:  # Even the subset yields no majority, we cannot continue
            break

        majority_class, _ = unpack_hypothesis_belief_pair(majority)
        taxonomy_list[i] = (majority_class, original_evaluation[(majority_class,)], original_evaluation)
        previous_class = majority_class

    return [(classification, belief) for classification, belief, evaluation in taxonomy_list]


def merge_taxonomy_classes(taxonomy, evaluation: MassFunction) -> MassFunction:
    """Move one level up the taxonomy, merging classes to into their parent classes."""
    merged_evaluation = MassFunction()
    for classification_hypothesis, belief in evaluation.pignistic().items():
        classification = list(classification_hypothesis)[0]  # Unfreeze the frozenset
        parent = taxonomy.get_parent(classification)
        merged_evaluation[(parent,)] += belief
    return merged_evaluation.normalize()
