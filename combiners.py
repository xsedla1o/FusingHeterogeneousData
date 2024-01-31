"""
Module implementing basic combination methods.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
import sys
from collections import defaultdict

import pandas as pd
from sklearn.metrics import accuracy_score

import data_io

args = data_io.parser.parse_args()

config = data_io.get_config(args.config)
module_config = data_io.get_config_section(config, args)

# Load config
PRIORITY_OUT = data_io.get_config_item(module_config, "PRIORITY_OUT")
MAJORITY_OUT = data_io.get_config_item(module_config, "MAJORITY_OUT")
ORACLE_OUT = data_io.get_config_item(module_config, "ORACLE_OUT")

# Load data
result_df_1 = data_io.load_results(module_config, "LABEL_RESULTS_SOURCE")
joined_reference = data_io.load_reference(module_config, "REFERENCE_SOURCE", result_df_1)


# Oracle combiner
oracle_results = pd.Series(index=result_df_1.index, dtype=object)
for i, row in result_df_1.iterrows():
    best = ""
    for key, val in row.items():
        if key == "http_ua":
            continue
        if val == joined_reference.loc[i]:
            best = val
            break
        if val != "":
            best = val
    oracle_results[i] = best

oracle_results.to_csv(ORACLE_OUT)
print("Oracle done.", file=sys.stderr)

# Priority combiner
module_acc = {}
for module in result_df_1.columns:
    ref = joined_reference[result_df_1[module] != ""]
    pred = result_df_1[result_df_1[module] != ""][module]
    module_acc[module] = accuracy_score(ref, pred)
if "http_ua" in module_acc:
    del module_acc["http_ua"]
print(module_acc)

priority_ordered_modules = [x[0] for x in sorted(list(module_acc.items()), key=lambda x: x[1], reverse=True)]
print(priority_ordered_modules)
prioritized_results = pd.Series(index=result_df_1.index, dtype=object)
for i, row in result_df_1.iterrows():
    for name in priority_ordered_modules:
        if row[name] != "":
            prioritized_results[i] = row[name]
            break
    else:
        prioritized_results[i] = ""

prioritized_results.to_csv(PRIORITY_OUT)
print("Prioritized done.", file=sys.stderr)

# Majority combiner
majority_results = pd.Series(index=result_df_1.index, dtype=object)
for i, row in result_df_1.iterrows():
    result: dict[str, float] = defaultdict(float)
    for module, weight in module_acc.items():
        if row[module] != "":
            result[row[module]] += weight

    if len(result) == 0:
        majority_results[i] = ""
    else:
        majority_results[i] = sorted(list(result.items()), key=lambda x: x[1], reverse=True)[0][0]

majority_results.to_csv(MAJORITY_OUT)
print("Majority voting done.", file=sys.stderr)
