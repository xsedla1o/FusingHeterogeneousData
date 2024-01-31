"""Module for generating plots and tables to be used in Latex report."""
import json
import os
import shutil
from collections import defaultdict
from math import sqrt
from typing import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import data_io
from characteristic_extractor import fill_distribution
from constants import generic_labels
from filter import count_concurrent_sources
from rule_training import function_select

# Data structure configuration
with open("config/experiments.yml") as in_f:
    config = yaml.safe_load(in_f)
ATTR_COLUMNS = data_io.get_config_item(config, "ATTR_COLUMNS")

output_folder = "latex/generated/"
cmap = "Greens"
MODULE_N = 10

Path(output_folder).mkdir(parents=True, exist_ok=True)


def plot_intersection_statistics(summary: pd.DataFrame):
    """Plot heatmap matrix with intersection statistics."""
    intersections = pd.DataFrame(index=ATTR_COLUMNS, columns=ATTR_COLUMNS, dtype=float)
    for col1 in ATTR_COLUMNS:
        for col2 in ATTR_COLUMNS:
            intersect = summary[
                (summary[col1].notnull())
                & (summary[col1] != "[]")
                & (summary[col2].notnull())
                & (summary[col2] != "[]")
            ]
            intersections.loc[col1, col2] = len(intersect.index)
            if col1 == col2:
                break
    intersections = intersections.fillna(0.0)

    mask = np.zeros_like(intersections, dtype=bool)
    mask[np.triu_indices_from(mask, 1)] = True
    plt.figure(figsize=(12, 12))
    sn.heatmap(intersections, annot=True, mask=mask, fmt="g", cmap=cmap, cbar=False, square=True)
    plt.savefig(os.path.join(output_folder, "intersection.pdf"), bbox_inches="tight")


# Visualization & metrics function
short_labels = [
    "Windows",
    "Linux",
    "MacOS",
    "iOS",
    "Android",
]


def show_cm_and_metrics(ref_true, predicted, labels, title="Confusion Matrix", outfilename="cm.pdf", collect=None):
    """Plot a confusion matrix for given data, also save metrics to `collect`."""
    cm = confusion_matrix(ref_true, predicted, labels=labels)
    cm_df = pd.DataFrame(cm, index=short_labels, columns=short_labels)

    plt.figure(figsize=(5, 5))
    sn.heatmap(cm_df, annot=True, fmt="g", cmap=cmap, cbar=False, square=True)
    plt.title(title)
    plt.ylabel("Real values")
    plt.xlabel("Predicted values")

    plt.savefig(os.path.join(output_folder, outfilename), bbox_inches="tight")

    p = precision_score(ref_true, predicted, average="weighted", zero_division=0)
    r = recall_score(ref_true, predicted, average="weighted", zero_division=0)
    f1 = f1_score(ref_true, predicted, average="weighted", zero_division=0)
    a = accuracy_score(ref_true, predicted)

    if collect is not None:
        collect["Precision"] = p
        collect["Recall"] = r
        collect["F1 Score"] = f1
        collect["Accuracy"] = a


def str_formatter(v):
    """Format the given value and escape Latex-unsafe characters."""
    if isinstance(v, str):
        for char in ["\\", "&", "%", "$", "#", "{", "}", "~", "^"]:
            if char in v:
                v = v.replace(char, f"\\{char}")
        return v
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        if int(v) != v:
            return f"{v:.3f}"
    return str(v)


def bold(v):
    return f"\\textbf{{{v}}}"


def dataframe_to_latex(df, bold_max=None):
    """Transform given df to a Latex table."""
    table = [
        "\t\\begin{tabular}" f"{{ l {' '.join(['r' for _ in df.columns])} }}\n\t\t\\hline",
        f"\t\tIndex & {' & '.join([str_formatter(x) for x in df.columns])}\\\\",
        "\t\t\\hline",
    ]
    max_i = -1
    if bold_max is not None:
        max_i = df[bold_max].idxmax()

    for df_i, df_row in df.iterrows():
        if df_i == max_i:
            table.append(f"\t\t{bold(df_i)} & {' & '.join([bold(str_formatter(x[1])) for x in df_row.items()])}\\\\")
        else:
            table.append(f"\t\t{df_i} & {' & '.join([str_formatter(x[1]) for x in df_row.items()])}\\\\")
    table.append("\t\t\\hline\n\t\\end{tabular}\n")
    string_table = "\n".join(table)
    return string_table.replace("_", "\\_")


def series_to_latex(series):
    """Transform a given series to a Latex table."""
    table = [
        f"\t\\begin{{tabular}}" f"{{ l {' '.join(['r' for _ in series.index])} }}\n\t\t\\hline",
        f"\t\tIndex & {' & '.join([str_formatter(x) for x in series.index])}\\\\",
        "\t\t\\hline",
        f"\t\t & {' & '.join([str_formatter(x) for x in series])}\\\\",
        "\t\t\\hline\n\t\\end{tabular}\n",
    ]
    string_table = "\n".join(table)
    return string_table.replace("_", "\\_")


####################
# Dataset Statistics
####################

# Loading dataset
input_df = pd.read_csv("data/generated/sanitized_input.csv")

# show stats
plot_intersection_statistics(input_df)

# Fill NaN with None
input_df = input_df.where(pd.notnull(input_df), None)
input_df = input_df.where(pd.notnull(input_df), None)

result_df_1 = pd.read_csv("data/generated/sanitized_results.csv").drop(columns=["Unnamed: 0"])
result_df_1.index = input_df.index
result_df_1 = result_df_1.where(pd.notnull(result_df_1), "")

active_counts = count_concurrent_sources(result_df_1)
module_counts = pd.DataFrame(index=result_df_1.columns, columns=range(1, active_counts.max() + 1), dtype=int)
for col in module_counts.columns:
    selected = result_df_1[active_counts == col]
    for module in module_counts.index:
        module_counts.loc[module, col] = selected[selected[module] != ""].shape[0]

for n in module_counts.columns:
    module_counts[n] = module_counts[n].astype("int64")

with open(os.path.join(output_folder, "module_table.tex"), "w") as out:
    out.write(dataframe_to_latex(module_counts))

module_counts_ttl = module_counts.T.loc[:, module_counts.index != "http_ua"].T.sum()

for index, count in module_counts_ttl.items():
    module_counts_ttl[index] = count / index

module_counts_df = pd.DataFrame(
    {
        "sum": module_counts_ttl,
        "ratio": module_counts_ttl / module_counts_ttl.sum(),
    }
).T

with open(os.path.join(output_folder, "module_table_ttl.tex"), "w") as out:
    out.write(dataframe_to_latex(module_counts_df))

joined_generic = pd.read_csv("data/generated/sanitized_joined_results.csv")["0"]
joined_generic = joined_generic.fillna("")
joined_generic.index = input_df.index

os_distribution = pd.DataFrame(
    index=generic_labels,
    columns=range(1, active_counts.max() + 1),
    dtype=int,
)
os_distribution = fill_distribution(active_counts, joined_generic, joined_generic, os_distribution)

with open(os.path.join(output_folder, "os_table.tex"), "w") as out:
    out.write(dataframe_to_latex(os_distribution))

joined_reference = pd.read_csv("data/generated/sanitized_reference.csv").drop(columns=["Unnamed: 0"]).results

os_distribution = pd.DataFrame(
    index=generic_labels,
    columns=range(1, active_counts.max() + 1),
    dtype=int,
)
os_distribution = fill_distribution(active_counts, joined_reference, joined_generic, os_distribution)

with open(os.path.join(output_folder, "os_table_reference.tex"), "w") as out:
    out.write(dataframe_to_latex(os_distribution))

module_counts_index = {n: f"N={n}" for n in range(1, MODULE_N)}
module_counts_index_with_complete = {"sum": "Entire Dataset", **module_counts_index}  # type: ignore

plot_module_counts = module_counts.drop(labels="http_ua")
plot_module_counts["Entire Dataset"] = plot_module_counts.T.sum()
plot_module_counts = plot_module_counts.T
plot_module_counts = plot_module_counts.rename(
    columns={
        "os_by_tcpip": "OS by TCP/IP",
        "os_by_tls": "TLS Fingerprinting",
        "sdp_labels": "SDP Analyzer",
        "tags_by_services": "Service Labels",
    },
    index=module_counts_index,
).reindex(list(module_counts_index_with_complete.values()))

divider_series = module_counts.drop(labels="http_ua").sum()
for i, total in divider_series.items():
    divider_series[i] = total / i
divider_series["sum"] = divider_series.sum()
divider_series = divider_series.rename(module_counts_index_with_complete)

#######################
# Experiment Evaluation
###############################
# 1. Testing individual modules

attr_to_module_name = {
    "http_ua": "HTTP UA",
    "os_by_tcpip": "OS by TCP/IP",
    "os_by_tls": "TLS Fingerprinting",
    "sdp_labels": "SDP Analyzer",
    "tags_by_services": "Service Labels",
}
metrics = ["Precision", "Recall", "F1 Score", "Accuracy"]
module_results_df = pd.DataFrame(index=metrics, columns=result_df_1.columns, dtype="float64")
module_results_df.drop(columns="http_ua", inplace=True)
filenames = []
for col in result_df_1.columns:
    if col == "http_ua":
        continue
    cut_results = result_df_1[result_df_1[col] != ""]
    reference = joined_reference[result_df_1[col] != ""]
    filenames.append(f"cm-{col}.pdf")
    show_cm_and_metrics(
        reference,
        cut_results[col],
        generic_labels,
        title=f"{attr_to_module_name.get(col, col)} Module",
        outfilename=f"cm-{col}.pdf",
        collect=module_results_df[col],
    )

with open(os.path.join(output_folder, "individual_modules_table.tex"), "w") as out:
    out.write(dataframe_to_latex(module_results_df))

# Generate latex code to present the CMs
plot_cnt = int(sqrt(len(filenames)))
plot_size = (1 / plot_cnt) - 0.01
output_lines = []

for i, filename in enumerate(filenames):
    output_lines.append(f"\\includegraphics[width={plot_size:.2f}\\linewidth]{{{filename}}}")
    output_lines.append("\\\\[1pt]" if (i + 1) % plot_cnt == 0 else "\\hfill")
    output_lines.append("\n")

output_string = "".join(output_lines)
with open(os.path.join(output_folder, "individual_modules_plots.tex"), "w") as out:
    out.write(output_string)

####################################
# 2. Module-based information fusion

module_fusion_df = pd.DataFrame(index=metrics, columns=["D-S"], dtype="float64")
joined_generic = pd.read_csv("data/generated/sanitized_joined_results.csv")["0"]
joined_generic = joined_generic.fillna("")
cut_joined_generic = joined_generic[joined_generic != ""]
cut_reference = joined_reference[joined_generic != ""]

show_cm_and_metrics(
    cut_reference,
    cut_joined_generic,
    generic_labels,
    title="D-S theory",
    outfilename="cm-informed-dst.pdf",
    collect=module_fusion_df["D-S"],
)

###################
# 3. Training rules

ds1_and_extras = pd.read_csv("data/generated/kfold_results-ds1.csv", index_col=0).rename(
    columns={"DS_teorie": "D-S1", "Prioritni": "PV", "Vetsinove": "WMV", "Oracle": "ORA"},
).drop(columns=["DS#_teorie", "PV"])
ds2 = pd.read_csv("data/generated/kfold_results-ds2.csv", index_col=0)
ds1_and_extras.insert(1, "D-S2", ds2["DS_teorie"])
ds1_and_extras.insert(3, "DSGD", ds2["DS#_teorie"])
ds1_and_extras.insert(0, "", metrics)
ds1_and_extras = ds1_and_extras.set_index("")

with open(os.path.join(output_folder, "training_rules_table.tex"), "w") as out:
    out.write(dataframe_to_latex(ds1_and_extras))

norm_trained_df = ds1_and_extras.copy()
for col in norm_trained_df:
    norm_trained_df[col] /= norm_trained_df["ORA"]
norm_trained_df.drop(columns=["ORA"], inplace=True)

with open(os.path.join(output_folder, "training_rules_table_norm.tex"), "w") as out:
    out.write(dataframe_to_latex(norm_trained_df))

#####################################
# 4. Modulating belief distributions

separated_functions: dict[str, dict[str, Callable]] = {fn_name: {} for fn_name in function_select}
for file in [
    "data/generated/optimized_params.csv",
]:
    params = pd.read_csv(file, index_col=0).T
    for fn_name, f in function_select.items():
        params_col = params[fn_name]
        separated_functions[fn_name][f"{fn_name}({params_col['a']:.2f}, {params_col['b']:.2f})"] = f(
            params_col["a"], params_col["b"]
        )

x = np.arange(0, 1, 0.01)

pd.DataFrame({fn_name: [f(c) for c in x] for fn_name, f in separated_functions["f1"].items()}, index=x).plot(
    colormap="Paired", xlabel="x", ylabel="y", figsize=(5, 5), ylim=(-0.05, 1.05)
)
plt.savefig(os.path.join(output_folder, "f1.pdf"), bbox_inches="tight")

pd.DataFrame({fn_name: [f(c) for c in x] for fn_name, f in separated_functions["f2"].items()}, index=x).plot(
    colormap="Paired", xlabel="x", ylabel="y", figsize=(5, 5), ylim=(-0.05, 1.05)
)
plt.savefig(os.path.join(output_folder, "f2.pdf"), bbox_inches="tight")

modulating_df = pd.read_csv("data/generated/param_results.csv", index_col=0).rename(
    columns={"precision": "Precision", "recall": "Recall", "f1": "F1 Score", "accuracy": "Accuracy"}
)

with open(os.path.join(output_folder, "modulating_belief_table.tex"), "w") as out:
    out.write(dataframe_to_latex(modulating_df, bold_max="F1 Score"))

################
# Final analysis
################

metrics_dict = defaultdict(list)

complete_df = pd.read_csv("data/generated/sanitized_results.csv").drop(columns=["Unnamed: 0", "http_ua"]).fillna("")
complete_ref_df = pd.read_csv("data/generated/sanitized_reference.csv").drop(columns=["Unnamed: 0"]).results
for col in complete_df.columns:
    metrics_dict[col].append(
        accuracy_score(complete_ref_df[complete_df[col] != ""], complete_df[complete_df[col] != ""][col])
    )

for n in range(1, MODULE_N):
    selector = count_concurrent_sources(result_df_1) == n
    filtered_df = complete_df[selector].reset_index().drop(columns=["index"])
    filtered_ref = joined_reference[selector].reset_index().drop(columns=["index"])
    print(n, filtered_df.shape, filtered_ref.shape)
    if filtered_df.shape[0] == 0:
        for col in filtered_df.columns:
            metrics_dict[col].append(np.nan)
        continue
    for col in filtered_df.columns:
        cut_col = filtered_df[filtered_df[col] != ""][col]
        if len(cut_col) == 0:
            metrics_dict[col].append(np.nan)
            continue
        metrics_dict[col].append(accuracy_score(filtered_ref[filtered_df[col] != ""], cut_col))

plot_labels = list(module_counts_index_with_complete.values())
metrics_df = pd.DataFrame(metrics_dict, index=plot_labels).rename(
    columns={
        "os_by_tcpip": "OS by TCP/IP",
        "os_by_tls": "TLS Fingerprinting",
        "sdp_labels": "SDP Analyzer",
        "tags_by_services": "Service Labels",
    }
)

# Modules accuracy plot
pd.DataFrame(metrics_df, index=plot_labels).plot(
    style=["s-" for _ in metrics_df.columns], title="Přesnost modulů", figsize=(5, 4), ylabel="Přesnost"
)
plt.savefig(os.path.join(output_folder, "modules-acc.pdf"), bbox_inches="tight")

# Modules accuracy plot
counts_df = plot_module_counts.copy()

for label in divider_series.index:
    counts_df.loc[label] /= divider_series[label]

(counts_df * metrics_df).plot(
    style=["s-" for _ in metrics_df.columns],
    figsize=(5, 4),
    title="Accuracy weighted by ratio in dataset",
    ylabel="Accuracy",
)
plt.savefig(os.path.join(output_folder, "modules_acc_weighted.pdf"), bbox_inches="tight")

# Method metrics plot
summary_df = module_fusion_df.join(ds1_and_extras)
output_columns = [
    "WMV",
    "D-S1",
    "D-S2",
    "DSGD",
    "ORA",
    "DT",
    "RF",
    "AB",
]
output_colors = [
    "tab:orange",
    "tab:purple",
    "tab:pink",
    "tab:olive",
    "tab:gray",
    "salmon",
    "darkred",
    "deeppink",
]

sorted_columns = sorted(
    (
        (col, color, summary_df[col].sum())
        for col, color in zip(output_columns, output_colors)
        if summary_df.loc["F1 Score", col] > 0.5
    ),
    key=lambda y: y[2],
)
sorted_output_columns = [x[0] for x in sorted_columns]
sorted_colors = [x[1] for x in sorted_columns]
boundary_df = summary_df[[col for col in summary_df.columns if summary_df.loc["F1 Score", col] > 0.5]].loc[
    ["F1 Score", "Accuracy"], :
]
min_val = boundary_df.min().min()
max_val = boundary_df.max().max()

ax = summary_df.loc[["Accuracy", "F1 Score"], sorted_output_columns].plot(
    kind="bar",
    figsize=(4, 2.5),
    ylim=(max(min_val, 0), min(max_val, 1)),
    color=sorted_colors,
)
ax.grid(True, axis="y", linestyle="dotted", linewidth=0.25)
ax.legend(bbox_to_anchor=(1, 0.9))
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(np.arange(round(min_val, 1) - 0.05, min(round(max_val, 1) + 0.1, 1), 0.05))
plt.savefig(os.path.join(output_folder, "methods_metrics.pdf"), bbox_inches="tight")

##############
# Rule listing
##############

with open("data/generated/out.conf") as rule_file, open(os.path.join(output_folder, "rules.tex"), "w") as tex_file:
    tex_file.write("\\begin{minted}[linenos, breaklines, tabsize=2]{python}\n")
    tex_file.write(rule_file.read())
    tex_file.write("\n\\end{minted}")
