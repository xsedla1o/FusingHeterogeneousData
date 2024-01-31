"""
Implementation of classes managing the class taxonomy.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from yaml import safe_load as safe_load_yaml


@dataclass
class ClassificationTaxonomyNode:
    """Represents a single node in the classification taxonomy."""

    parent: ClassificationTaxonomyNode | None = field(repr=False)
    value: str
    children: list[ClassificationTaxonomyNode] | None


def load_classification_taxonomy(taxonomy_file_path: str = None, forbidden_chars: str = None) -> dict:
    """
    Loads a classification taxonomy dictionary from the specified YAML file.

    The YAML file should contain a single dictionary of dictionaries, named Device.
    Leaf categories should be represented with null values.

    :param taxonomy_file_path: Path to YAML file, expected structure is dict of dicts.
        If None is passed, the default config will be used.
    :param forbidden_chars: Characters that must not appear in class names.(default='. ')
    :return: taxonomy dict
    """
    if taxonomy_file_path is None:
        current_dir = os.path.dirname(__file__)
        taxonomy_file_path = os.path.join(current_dir, "data", "label_fusion_taxonomy.yaml")

    with open(taxonomy_file_path) as source:
        taxonomy = safe_load_yaml(source)

    forbidden_chars = ". " + forbidden_chars if forbidden_chars is not None else ". "
    validate_taxonomy(taxonomy, forbidden_chars)

    for sub_taxonomy in taxonomy.values():
        even_out_taxonomy_depth(sub_taxonomy)
        validate_taxonomy_depth(sub_taxonomy)

    return taxonomy


def validate_taxonomy(taxonomy: dict | None, forbidden_chars: str) -> None:
    """Validate taxonomy format: nested dicts with string keys that do not contain forbidden chars."""
    if taxonomy is None:
        return

    if not isinstance(taxonomy, dict):
        raise ValueError("Taxonomy format invalid, expected dict of dicts.")

    for key, value in taxonomy.items():
        if isinstance(key, str):
            for char in forbidden_chars:
                if char in key:
                    raise SyntaxError(f"Use of character '{char}' is forbidden - found in '{key}' ({forbidden_chars=})")
        else:
            raise TypeError("Taxonomy format invalid, expected only string keys.")
        validate_taxonomy(value, forbidden_chars)


def even_out_taxonomy_depth(taxonomy: dict, generic_name="_") -> None:
    max_depth = get_taxonomy_max_depth(taxonomy)
    extend_taxonomy_by_generic(taxonomy, generic_name, max_depth)


def get_taxonomy_max_depth(taxonomy: dict | None) -> int:
    if taxonomy is None:
        return 0
    return max(get_taxonomy_max_depth(value) for value in taxonomy.values()) + 1


def extend_taxonomy_by_generic(taxonomy: dict, generic_name: str, remaining_depth: int) -> None:
    """Extend taxonomy by generic subclasses to reach even depth in all branches."""
    remaining_depth -= 1
    if remaining_depth == 0:
        return
    for key, value in taxonomy.items():
        if value is None:
            taxonomy[key] = {generic_name: None}
        extend_taxonomy_by_generic(taxonomy[key], generic_name, remaining_depth)


def validate_taxonomy_depth(taxonomy: dict | None) -> int:
    """Validate that taxonomy depth is even in all branches."""
    if taxonomy is None:
        return 0
    branch_depth: int | None = None
    for value in taxonomy.values():
        if branch_depth is None:
            branch_depth = validate_taxonomy_depth(value)
        elif branch_depth != validate_taxonomy_depth(value):
            break
    else:
        assert isinstance(branch_depth, int)
        return branch_depth + 1
    raise ValueError("Taxonomy format invalid, uneven taxonomy tree depths.")


class ClassificationTaxonomyManager:
    """Class that enables using the classification taxonomy and provides helper functions."""

    def __init__(self, class_taxonomy: dict = None):
        if class_taxonomy is None:
            class_taxonomy = load_classification_taxonomy()

        self.root: ClassificationTaxonomyNode
        self.class_map: dict[str, ClassificationTaxonomyNode] = {}
        self.root_map = {}
        for root, taxonomy in class_taxonomy.items():
            self.root_map[root] = self._construct_taxonomy_tree(taxonomy, root)

        self.enum = Enum("Classes", {v.lower(): v for v in self.class_map})  # type: ignore
        self.enum.__str__ = lambda s: s.value
        self.enum.__repr__ = lambda s: s.value

    def _construct_taxonomy_tree(
        self, class_taxonomy: dict, value: str, parent: ClassificationTaxonomyNode = None
    ) -> ClassificationTaxonomyNode:
        if parent is not None:
            value = parent.value + "." + value

        if class_taxonomy is None:
            node = ClassificationTaxonomyNode(parent, value, None)
            self.class_map[value] = node
            return node

        node = ClassificationTaxonomyNode(parent, value, [])
        self.class_map[value] = node

        for classification, descendants in class_taxonomy.items():
            assert node.children is not None
            node.children.append(self._construct_taxonomy_tree(descendants, classification, node))
        return node

    def _get_leaf_class_str_list(self, node: ClassificationTaxonomyNode) -> list[str]:
        """Returns a list of leaf classifications for a specified node. (root if None is passed)"""
        leaf_classes: list[str] = []
        self._get_leaf_classes_recursive(node, leaf_classes)
        return leaf_classes

    def _get_leaf_classes_recursive(self, node: ClassificationTaxonomyNode, leaf_classes: list) -> None:
        if node.children is None:
            leaf_classes.append(node.value)
            return

        for child in node.children:
            self._get_leaf_classes_recursive(child, leaf_classes)

    def create_class_list(self, classification: str | Enum) -> list[Enum]:
        """Returns a list of classification Enum instances for the given classification."""
        if isinstance(classification, Enum):
            class_node = self.class_map[classification.value]
        else:
            class_node = self.class_map[classification]
        leaf_class_list = self._get_leaf_class_str_list(class_node)
        return [self.enum(item) for item in leaf_class_list]

    def get_all_classes_of_group(self, classification: Enum) -> tuple[Enum, ...]:
        """Returns a list of all possible classifications that belong to the same group as the one provided."""
        tree_root = self.get_root_class_of(classification)
        return tuple(self.create_class_list(tree_root))

    @staticmethod
    def get_root_class_of(classification: Enum):
        """Returns the root classification of the one provided."""
        classification_tree_path = classification.value.split(".")
        return classification_tree_path[0]

    def _get_node(self, value: str | Enum):
        if isinstance(value, Enum):
            return self.class_map[value.value]
        return self.class_map[value]

    def get_parent(self, classification: str | Enum) -> Enum:
        """Returns a parent classification for the one passed as argument."""
        node = self._get_node(classification)
        assert node.parent is not None
        return self.enum(node.parent.value)

    def get_children(self, classification: str | Enum) -> list[Enum]:
        """Returns children classifications of the one passed as argument."""
        node = self._get_node(classification)
        assert node.children is not None
        return [self.enum(item.value) for item in node.children]
