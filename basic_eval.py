"""
Module implementing the initial classification of the dataset.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
import ast
from operator import itemgetter
from typing import Optional

import pandas as pd
from dp3.common.attrconvert import is_iterable
from dp3.common.config import load_attr_spec, read_config_dir

import data_io
from label_fusion_classification import merge_taxonomy_classes
from pyds.pyds import MassFunction
from ruleloader.classification_taxonomy import (
    ClassificationTaxonomyManager,
    load_classification_taxonomy,
)
from ruleloader.dp3ifc import DP3Ifc
from ruleloader.ruleloader import RuleLoader
from ruleloader.rules import OfflineRuleFactory

TAXONOMY_FILE_PATH = "config/label_fusion_taxonomy.yaml"

# Creating all classification objects
taxonomy = ClassificationTaxonomyManager(load_classification_taxonomy(TAXONOMY_FILE_PATH))


def raw_to_generic_topclass(raw: list):
    if raw is None or len(raw) == 0:
        return ""
    return sorted(
        (
            (tuple(clss)[0], bel)
            for clss, bel in merge_taxonomy_classes(taxonomy, MassFunction(raw)).pignistic().items()
        ),
        key=itemgetter(1),
        reverse=True,
    )[0][0].value


def convert_dataset_item(value: str, attr_attr_spec):
    """Return item converted to python object, or the original string on fail."""
    if attr_attr_spec.multi_value or is_iterable(attr_attr_spec.data_type):
        try:
            return ast.literal_eval(value)
        except ValueError as ex:
            print(ex)
            print(value)
            return None
    else:
        return value


def apply_rules(rule_list, device_profile):
    """Apply all rules in `rule_list` onto `device_profile`."""
    res = None
    for rule in rule_list:
        if res is None:
            res = rule(device_profile)
        else:
            tmp = rule(device_profile)
            if tmp is not None:
                res &= tmp
    return res


if __name__ == "__main__":
    args = data_io.parser.parse_args()

    config = data_io.get_config(args.config)
    module_config = data_io.get_config_section(config, args)

    # Load config
    ATTR_CONF_DIR = data_io.get_config_item(config, "ATTR_CONF_DIR")
    LABEL_RESULTS_OUTPUT = data_io.get_config_item(module_config, "LABEL_RESULTS_OUTPUT")
    JOINED_RESULTS_OUTPUT = data_io.get_config_item(module_config, "JOINED_RESULTS_OUTPUT")
    evaluation_rule_sets = data_io.get_config_item(module_config, "RULES")
    print(evaluation_rule_sets)

    # Load data
    attr_spec = data_io.load_attr_spec(config)
    ATTR_COLUMNS = data_io.get_config_item(config, "ATTR_COLUMNS")
    input_df = data_io.load_raw_data(module_config, "RAW_INPUT_SOURCE")

    # This runs classification of the entire dataset
    raw_results = pd.DataFrame(index=input_df.index, columns=evaluation_rule_sets.keys())
    dp3ifc = DP3Ifc(load_attr_spec(read_config_dir(ATTR_CONF_DIR)))
    loader = RuleLoader(rule_factory=OfflineRuleFactory(), dp3_ifc=dp3ifc, class_taxonomy=taxonomy)

    for colname, rulefile in evaluation_rule_sets.items():
        rules = loader.parse_file(rulefile)

        for i, row in input_df.iterrows():
            profile: dict[str, Optional[list]] = {
                attr: [] if attr_spec[attr].multi_value else None for attr in ATTR_COLUMNS
            }
            for key, item in row.items():
                if key in ATTR_COLUMNS and item is not None:
                    profile[key] = convert_dataset_item(item, attr_spec[key])
            result = apply_rules(rules, profile)

            if result is not None:
                raw_results.loc[i, colname] = list(result.items())

    raw_results = raw_results.where(pd.notnull(raw_results), None)

    # Output results of base modules as labels.
    result_df_1 = raw_results.applymap(raw_to_generic_topclass)
    result_df_1.to_csv(LABEL_RESULTS_OUTPUT)

    # Join raw results using the basic rules
    joined_results = pd.Series(index=raw_results.index, dtype=object)
    for i, row in raw_results.loc[:, raw_results.columns != "http_ua"].iterrows():
        maf = None
        for key, val in row.items():
            if val is not None:
                if maf is None:
                    maf = MassFunction(val)
                else:
                    maf &= MassFunction(val)
        joined_results[i] = list(maf.items()) if maf is not None else None

    joined_generic = joined_results.apply(raw_to_generic_topclass)
    joined_generic.to_csv(JOINED_RESULTS_OUTPUT)
