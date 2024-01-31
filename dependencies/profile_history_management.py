"""Module aggregating functions for managing profile histories and accessing values at certain times."""
from datetime import timedelta
from operator import itemgetter

import numba

from dependencies.datapointloader import AttrSpec, get_attr_spec


def attr_has_history(attr_name: str):
    sel_attr_spec = get_attr_spec(attr_name)
    return sel_attr_spec.history


@numba.njit
def get_conf_at_t_fast(base_conf, time, t1, t2, pre_val, post_val):  # pylint: disable=R0913
    """Get the confidence value at given time, numba accelerated."""
    if time < t1:
        if time <= t1 - pre_val:
            return 0.0
        return base_conf * (1 - (t1 - time) / pre_val)
    if time <= t2:
        return base_conf  # completely inside the (strict) interval
    if time >= t2 + post_val:
        return 0.0
    return base_conf * (1 - (time - t2) / post_val)


def get_conf_at_t(base_conf, time, t1, t2, pre_val, post_val):  # pylint: disable=R0913
    """Get the confidence value at given time."""
    if time < t1:
        if time <= t1 - pre_val:
            return 0.0
        return base_conf * (1 - (t1 - time) / pre_val)
    if time <= t2:
        return base_conf  # completely inside the (strict) interval
    if time >= t2 + post_val:
        return 0.0
    return base_conf * (1 - (time - t2) / post_val)


def get_current_value(attr_spec: AttrSpec, value_history, time=None):
    """
    Get most probable value (and its confidence) of given attribute at given time.

    Or list of all possible values (and corresponding confidences) for multi_value attributes.

    :param attr_spec Specification of the attribute as returned by dp3.common.config.load_attr_spec
    :param value_history List/array of attribute values in time (t1,t2,c,val)
    :param time Queried time (datetime), if not specified, set to the maximum of t2 (latest value)

    :return 2-tuple with attribute value and its confidence, or a list of such tuples if attr_spec.multi_value=True
    """
    assert attr_spec.history is True
    if len(value_history) == 0:  # no input data
        return [] if attr_spec.multi_value else (None, 1.0)

    if time is None:
        time = max(row[1] for row in value_history)  # max of t2 over all rows

    pre_validity = attr_spec.history_params["pre_validity"]
    post_validity = attr_spec.history_params["post_validity"]

    # For each interval, compute confidence of the value at "time" (confidence is equal to "c" between t1 and t2, and
    # decreases linearly during pre-/post-validity intervals).
    # Store all values with non-zero confidence.
    values = []  # pairs of (value, computed confidence, t2)
    # (t2 is included to allow deterministic selection in case multiple values have the same confidence, see below)

    for t1, t2, c, val in value_history:
        conf = get_conf_at_t(c, time, t1, t2, pre_validity, post_validity)
        if conf > 0.0:
            values.append((val, conf, t2))

    # If attribute is multivalued, return all (value,conf) pairs (sorted by confidence)
    if attr_spec.multi_value:
        all_pairs = sorted(map(itemgetter(0, 1), values), key=itemgetter(1), reverse=True)
        # all_pairs includes duplicates, we keep only unique values with highest confidence
        included_values = []
        unique_value_pairs = []
        for val, conf in all_pairs:
            if val not in included_values:
                included_values.append(val)
                unique_value_pairs.append((val, conf))
        return unique_value_pairs

    if not values:  # there is no value with non-zero confidence
        return None, 1.0

    # Otherwise, find the one with the highest confidence. If there are multiple (may happen when time is exactly in
    # between two intervals, or when intervals are overlapping), select the one with the latest t2.
    res = max(values, key=itemgetter(1, 2))  # items 1,2 = conf,t2
    return res[0], res[1]  # (value, conf)


def get_value_at_time(profile: dict, attr_name: str, time=None, pre: float = 1.5, post: float = 3.0):
    """
    Get value of the attribute of the profile at the given time.

    :param profile: dict of attribute histories
    :param attr_name: name of the attribute
    :param time: time at which to get value, if unspecified, behavior is dependent on get_current_value()
    :param pre: pre_validity interval
    :param post: post_validity interval
    """
    sel_attr_spec = get_attr_spec(attr_name)

    if not sel_attr_spec.history:
        return profile[attr_name]

    sel_attr_spec.history_params["pre_validity"] = timedelta(hours=pre)
    sel_attr_spec.history_params["post_validity"] = timedelta(hours=post)

    return get_current_value(sel_attr_spec, profile[attr_name], time)


def get_current_values(profile, time=None, pre=2, post=4):
    """
    Get current value of all attributes of a profile.

    :param profile: dict of attribute histories
    :param time: time at which to get value, if unspecified, behavior is dependent on get_current_value()
    :param pre: pre_validity interval
    :param post: post_validity interval
    """
    values = {}
    for attribute in profile:
        current_value = get_value_at_time(profile, attribute, time, pre, post)
        if isinstance(current_value, tuple):
            values[attribute] = current_value[0]
        elif isinstance(current_value, list):
            values[attribute] = [x[0] for x in current_value]
        else:
            raise Exception(f"Something went wrong with current_value={current_value}, attribute={attribute}")
    return values


def get_profile_values_without_and_with_confidence(profile, time=None, pre=2, post=4):
    """
    Get current values of all attributes of a profile, with and without their respective confidence.

    :param profile: dict of attribute histories
    :param time: time at which to get value, if unspecified, behavior is dependent on get_current_value()
    :param pre: pre_validity interval
    :param post: post_validity interval
    """
    without, with_conf = {}, {}
    for attribute in profile:
        current_value = get_value_at_time(profile, attribute, time, pre, post)

        if attr_has_history(attribute):
            if isinstance(current_value, tuple):
                without[attribute] = current_value[0]
            elif isinstance(current_value, list):
                without[attribute] = [x[0] for x in current_value]
            else:
                raise Exception("Something went wrong")

            with_conf[attribute] = current_value
        else:
            with_conf[attribute] = current_value

            if isinstance(current_value, list):
                with_conf[attribute] = [(x, 1.0) for x in current_value]
            else:
                without[attribute] = current_value[0]

    return without, with_conf


def get_profile_values_without_and_with_confidence_preloaded(profile):
    """
    Get usable profile with current values, with and without their respective confidence.

    Args:
        profile: dict of attribute current values
    """
    without, with_conf = {}, {}
    for attribute, current_value in profile.items():
        if attr_has_history(attribute):
            if isinstance(current_value, tuple):
                without[attribute] = current_value[0]
            elif isinstance(current_value, list):
                without[attribute] = [x[0] for x in current_value]
            else:
                raise Exception("Something went wrong")

            with_conf[attribute] = current_value
        else:
            with_conf[attribute] = current_value

            if isinstance(current_value, list):
                with_conf[attribute] = [(x, 1.0) for x in current_value]
            else:
                without[attribute] = current_value[0]
    return without, with_conf


def get_profile_first_active_time(device_profile: dict):
    """Returns the first time a profile has been active."""
    first_active_time = None
    for attribute, value in device_profile.items():
        if not attr_has_history(attribute):
            continue
        t1, _, _, _ = value[0]
        if first_active_time is None or t1 < first_active_time:
            first_active_time = t1
    return first_active_time


def get_profile_last_active_time(device_profile: dict):
    """Returns the last time a profile has been active."""
    last_active_time = None
    for attribute, value in device_profile.items():
        if not attr_has_history(attribute):
            continue
        _, t2, _, _ = value[-1]
        if last_active_time is None or t2 > last_active_time:
            last_active_time = t2
    return last_active_time


def get_value_count(device_profile: dict, time, pre=2, post=4):
    """Returns the number of active values at given time."""
    values_at_time = get_current_values(device_profile, time, pre, post)
    value_count = 0
    for value in values_at_time.values():
        if isinstance(value, list):
            value_count += 1 if value else 0
        else:
            value_count += 1 if value is not None else 0
    return value_count
