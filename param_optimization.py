"""
Module implementing parameter optimization of functions.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import data_io
from basic_eval import convert_dataset_item
from constants import max_process_cnt
from rule_optimization_kfold import aggregate_rule_results
from rule_training import (
    function_select,
    get_normalized_confusion,
    get_replacement_mf_single_class_optimized,
    load_all_rules,
)
from ruleloader.rules import AbstractRangeRule


def find_optimal_params_inner(f__x__grid_steps):
    """
    Returns optimal parameters to optimized function f().
    Called by `find_optimal_params`, iterates over the y-axis.
    """
    function, x, grid_steps = f__x__grid_steps
    print(f"inner: function={function.__name__:20s} {x=}")
    best_score = 0.0
    best_params = None

    for y in grid_steps:
        tweaked_rules = deepcopy(all_rules)
        tweaked_results = pd.Series(index=input_df.index, dtype=object)

        fn = function(x, y)

        for rule_index in normalized_confusion_df:
            if isinstance(tweaked_rules[rule_index], AbstractRangeRule):
                continue
            tweaked_rules[rule_index].maf = get_replacement_mf_single_class_optimized(
                normalized_confusion_df[rule_index], all_classes, fn
            )

        get_tweaked_results(tweaked_results, tweaked_rules)

        score = f1_score(
            joined_reference[tweaked_results.notna()],
            tweaked_results[tweaked_results.notna()],
            average="weighted",
            zero_division=0,
        )
        if score > best_score:
            best_score = score
            best_params = (x, y)

    print(f"best found for: {function.__name__:20s} {x=:.2f}: {best_score=:.6f}, {best_params=}")
    return best_score, best_params


def get_tweaked_results(tweaked_results, tweaked_rules):
    """Classify dataset with tweaked rules and save results."""
    for i, row in input_df.iterrows():
        profile = {attr: [] if attr_spec[attr].multi_value else None for attr in ATTR_COLUMNS}
        for key, val in row.items():
            if key in ATTR_COLUMNS and val is not None:
                profile[key] = convert_dataset_item(val, attr_spec[key])
        aggregate_rule_results(profile, tweaked_rules, tweaked_results, i)


def find_optimal_params(fn_name__f):
    """
    Returns optimal parameters to optimized function f().
    Maps `find_optimal_params_inner` over the x-axis using multiprocessing.
    """
    print(f"outer, {fn_name__f}")
    fn_name, fn = fn_name__f
    grid_steps = np.arange(0, 1, 0.04)

    with ProcessPoolExecutor(max_process_cnt // 2) as inner_pool:
        x_results = inner_pool.map(find_optimal_params_inner, [(fn, x, grid_steps) for x in grid_steps])

    best_score, best_params = max(x_results, key=lambda x: x[0])

    print(f"{fn_name=}, {best_score=}, {best_params=}", file=sys.stderr)
    return fn_name, best_params


parser = data_io.parser
parser.add_argument("-r", "--reduce", type=int, help="Reduce the sample size to this number", default=None)
args = data_io.parser.parse_args()
config = data_io.get_config(args.config)
module_config = data_io.get_config_section(config, args)

# Data structure & paths configuration
PARAM_OUTPUT = data_io.get_config_item(module_config, "PARAM_OUTPUT")
RANDOM_STATE = data_io.get_config_item(config, "RANDOM_STATE")

# Load data
attr_spec = data_io.load_attr_spec(config)
ATTR_COLUMNS = data_io.get_config_item(config, "ATTR_COLUMNS")
input_df = data_io.load_raw_data(module_config, "RAW_INPUT_SOURCE")
joined_reference = data_io.load_reference(module_config, "REFERENCE_SOURCE", input_df)

if args.reduce is not None and input_df.shape[0] > args.reduce:
    selected_data = input_df.sample(args.reduce, random_state=RANDOM_STATE)
    selected = pd.Series(index=input_df.index, dtype=bool, data=[i in selected_data.index for i in input_df.index])

    input_df = input_df[selected].reset_index().drop(columns=["index"])
    joined_reference = joined_reference[selected].reset_index().drop(columns=["index"])

all_rules = load_all_rules(config)
all_classes = all_rules[0].all_classes

normalized_confusion_df = get_normalized_confusion(all_rules, input_df, joined_reference, attr_spec, ATTR_COLUMNS)
optimized_params_df = pd.DataFrame(index=function_select, columns=["a", "b"])

with ProcessPoolExecutor(min(max_process_cnt, len(function_select))) as pool:
    data_out = pool.map(find_optimal_params, function_select.items())

for item in data_out:
    function_name, params = item
    optimized_params_df.loc[function_name, "a"], optimized_params_df.loc[function_name, "b"] = params

optimized_params_df.to_csv(PARAM_OUTPUT)
