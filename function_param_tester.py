"""
Module testing optimized function parameters.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable

import pandas as pd
from sklearn.model_selection import KFold

import data_io
from constants import max_process_cnt
from rule_optimization_kfold import setup_distortion_resilience, test_rules, train_rules
from rule_training import function_select
from rule_training import get_replacement_mf_single_class_optimized as method

K = 5


def test_function_inner(i_split_indices, fn):
    """Test the rules on given indices."""
    i, split_indicies = i_split_indices
    train_index, test_index = split_indicies
    mod, a, b = fn
    exec_mod = mod(a, b)
    measured = {}

    trained_rules = train_rules(train_index, method, exec_mod, input_df, joined_reference, attr_spec, config)
    p, r, f1, a, res = test_rules(trained_rules, test_index, input_df, joined_reference, attr_spec, ATTR_COLUMNS)

    measured[(i, "precision")] = p
    measured[(i, "recall")] = r
    measured[(i, "f1")] = f1
    measured[(i, "accuracy")] = a
    measured[(i, "results")] = res

    return measured


def test_function(function_name__function):
    """Test passed function using the 5-fold cross validation."""
    function_name, function = function_name__function
    kf_out = pd.DataFrame(index=range(K), columns=["precision", "recall", "f1", "accuracy"])

    print(function_name)
    print(f"{'Precision':15s}|{'Recall':15s}|{'F1 score':15s}|{'Accuracy':15s}")
    print(f"{15 * '-'}|{15 * '-'}|{15 * '-'}|{15 * '-'}")

    with ProcessPoolExecutor(min(max_process_cnt // 2, K)) as inner_pool:
        metrics_out = inner_pool.map(partial(test_function_inner, fn=function), enumerate(split_indexes))

    results_s = pd.Series(index=input_df.index, dtype=object)
    for measured_results in metrics_out:
        for i_metric, metric_value in measured_results.items():
            i, metric_name = i_metric
            if metric_name == "results":
                results_s.fillna(metric_value, inplace=True)
                continue
            kf_out.loc[i, metric_name] = metric_value

    mean_res = kf_out.mean()
    return_val = {(function_name, metric_name): mean_res[metric_name] for metric_name in mean_res.index}
    return_val[(function_name, "results")] = results_s
    return return_val


data_io.parser.add_argument(
    "--distortion-resilience",
    action="store_true",
    help="whether to perform testing of resilience (requires special config)",
)
args = data_io.parser.parse_args()
config = data_io.get_config(args.config)
module_config = data_io.get_config_section(config, args)

# Data structure & paths configuration
ATTR_CONF_DIR = data_io.get_config_item(config, "ATTR_CONF_DIR")
RANDOM_STATE = data_io.get_config_item(config, "RANDOM_STATE")

PARAMS_SOURCES = data_io.get_config_item(module_config, "PARAMS_SOURCES")
METRICS_OUT = data_io.get_config_item(module_config, "METRICS_OUT")
RESULTS_OUT = data_io.get_config_item(module_config, "RESULTS_OUT")

# Load required data
attr_spec = data_io.load_attr_spec(config)
ATTR_COLUMNS = data_io.get_config_item(config, "ATTR_COLUMNS")

if args.distortion_resilience:
    input_df, joined_reference, split_indexes = setup_distortion_resilience(module_config, 5, RANDOM_STATE)
else:
    input_df = data_io.load_raw_data(module_config, "RAW_INPUT_SOURCE")
    joined_reference = data_io.load_reference(module_config, "REFERENCE_SOURCE", input_df)
    kf5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    split_indexes = kf5.split(input_df)

separated_functions: dict[str, dict[str, tuple[Callable, float, float]]] = {fn_name: {} for fn_name in function_select}
for file in PARAMS_SOURCES:
    params = pd.read_csv(file, index_col=0).T
    for fn_name, f in function_select.items():
        params_col = params[fn_name]
        separated_functions[fn_name][f"{fn_name}({params_col['a']:.2f}, {params_col['b']:.2f})"] = (
            f,
            params_col["a"],
            params_col["b"],
        )

functions = {fn_name: f for separated in separated_functions.values() for fn_name, f in separated.items()}

function_metrics = pd.DataFrame(index=functions, columns=["precision", "recall", "f1", "accuracy"])

with ProcessPoolExecutor(min(max_process_cnt, len(functions))) as pool:
    data_out = pool.map(test_function, functions.items())

results = pd.DataFrame(index=input_df.index)
for item in data_out:
    for fn_name_metric, value in item.items():
        fn_name, metric = fn_name_metric
        if metric == "results":
            results.insert(0, fn_name, value)
            continue
        function_metrics.loc[fn_name, metric] = value

results.to_csv(RESULTS_OUT)
function_metrics.to_csv(METRICS_OUT)
