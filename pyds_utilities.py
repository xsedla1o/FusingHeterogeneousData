"""
A set of utility functions for working with the PyDS library.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
import functools
from typing import Any, Callable

from pyds.pyds import MassFunction


def reduce(maf: MassFunction, condition: Callable[[frozenset, float], bool]) -> MassFunction:
    """
    Reduce the passed `maf` so that only the hypotheses and belief pairs that satisfy `condition` are kept.
    The returned `maf` is normalized.
    """
    new_mf = MassFunction()
    for hypothesis, belief in maf.items():
        if condition(hypothesis, belief):
            new_mf[hypothesis] = belief
    new_mf.normalize()
    return new_mf


def get_majority_hypothesis(maf: MassFunction) -> tuple[frozenset, float]:
    """Return the hypothesis, belief pair which has the majority of belief within the `maf`."""
    majority_list = [(x, y) for x, y in maf.pignistic().items() if y > 0.5]
    if len(majority_list) > 1:
        raise ValueError(*majority_list)
    return majority_list[0]


def unpack_hypothesis_belief_pair(belief_pair: tuple[frozenset, float]) -> tuple[Any, float]:
    """Defreezes the hypothesis inside frozenset. The hypothesis must have a single element."""
    hypothesis, belief = belief_pair
    assert len(hypothesis) == 1
    hypothesis = list(hypothesis)[0]
    return hypothesis, belief


def unite_evaluations(evaluations: list[MassFunction]) -> MassFunction:
    return functools.reduce(lambda x, y: x & y, evaluations)


def are_conflicting_classifications(classifications: list[MassFunction]) -> bool:
    """
    Return whether the passed list of `classifications`
    has any maf with completely different main hypothesis than the other mafs.
    """
    main_classification = None
    for classification in classifications:
        hypothesis = get_main_hypothesis(classification)
        main_classification = hypothesis if main_classification is None else main_classification
        if not (hypothesis & main_classification):
            return True
    return False


def get_main_hypothesis(evaluation: MassFunction) -> set:
    """Returns the main hypothesis. Two focal hypotheses `evaluation` maf is assumed."""
    for hypothesis in evaluation:
        # We can return the first item, since dictionaries maintain insertion order from 3.6 forward.
        return set(hypothesis)
    raise ValueError("Empty MassFunction")
