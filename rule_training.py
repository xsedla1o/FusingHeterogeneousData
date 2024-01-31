"""
Module implementing rule optimization.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
import sys
from collections import defaultdict
from copy import deepcopy
from operator import itemgetter

import pandas as pd
from dp3.common.config import load_attr_spec, read_config_dir
from sklearn.metrics import confusion_matrix

import data_io
from basic_eval import convert_dataset_item, raw_to_generic_topclass
from constants import generic_labels
from pyds.pyds import MassFunction
from ruleloader.classification_taxonomy import (
    ClassificationTaxonomyManager,
    load_classification_taxonomy,
)
from ruleloader.dp3ifc import DP3Ifc
from ruleloader.ruleloader import RuleLoader
from ruleloader.rules import AbstractBaseRule, OfflineRangeRule, OfflineRuleFactory

# Creating all classification objects
taxonomy = ClassificationTaxonomyManager(load_classification_taxonomy("config/label_fusion_taxonomy.yaml"))


def linear_step(p1, p2):
    """
    Belief modulation function.

    Equation: 1 / (`p2` - `p1`) * x - 1 / ((`p2` / `p1`) - 1)
    """
    if p1 == p2 or p1 == 0.0 or p2 < p1:  # p1 cant equal p2 and p1 cant equal 0
        return lambda _: 0

    def inner(x):
        if x < p1:
            return 0
        if x > p2:
            return 0.99
        return min(max(1 / (p2 - p1) * x - 1 / ((p2 / p1) - 1), 0.0), 0.99)

    return inner


def midpoint_transition(p1, p2):
    """
    Belief modulation function.

    Equations: x < `p1`: `p2` / `p1` * x;
    x >= `p1`: (`p2` - 1) / (`p1` - 1) * x + (`p1` - `p2`) / (`p1` - 1)
    """

    def inner(x):
        if x < p1:
            return p2 / p1 * x
        return (p2 - 1) / (p1 - 1) * x + (p1 - p2) / (p1 - 1)

    return inner


def get_replacement_mf(normalized_confusion_column, *_):
    """Return a mass function based on the `normalized_confusion_column`, all items are translated."""
    return MassFunction(
        {
            tuple(taxonomy.get_children(taxonomy.enum(os))): conf
            for os, conf in dict(normalized_confusion_column).items()
            if conf > 0.0
        }
    )


def get_replacement_mf_single_class(normalized_confusion_column, all_classes, *_):
    """
    Return a mass function based on the `normalized_confusion_column`,
    only the highest probability item is translated, the remaining mass is assigned to `all_classes`.
    """
    top_class, conf = max((key_val for key_val in dict(normalized_confusion_column).items()), key=itemgetter(1))
    conf = min(max(conf, 0.0), 0.99)
    return MassFunction({tuple(taxonomy.get_children(taxonomy.enum(top_class))): conf, all_classes: 1 - conf})


def get_replacement_mf_single_class_optimized(normalized_confusion_column, all_classes, mod):
    """
    Return a mass function based on the `normalized_confusion_column`,
    only the highest probability item is translated, the remaining mass is assigned to `all_classes`.
    On top of that, the belief assigned to the main hypothesis is first passed through `mod`.
    """
    top_class, conf = max((key_val for key_val in dict(normalized_confusion_column).items()), key=itemgetter(1))
    conf = min(max(mod(conf), 0.0), 0.99)
    return MassFunction({tuple(taxonomy.get_children(taxonomy.enum(top_class))): conf, all_classes: 1 - conf})


def load_all_rules(config_):
    """Load all rules based on passed `config`."""
    dp3ifc = DP3Ifc(load_attr_spec(read_config_dir(data_io.get_config_item(config_, "ATTR_CONF_DIR"))))
    loader = RuleLoader(
        rule_factory=OfflineRuleFactory(),
        dp3_ifc=dp3ifc,
        class_taxonomy=taxonomy,
    )
    all_rules = []
    evaluation_rule_sets = data_io.get_config_item(config_, "RULES")

    for colname, rulefile in evaluation_rule_sets.items():
        if colname == "http_ua":
            continue
        all_rules.extend(loader.parse_file(rulefile))
    return all_rules


def test_rule(i, row, rule, rule_index, rule_results, attr_spec_, attr_columns):
    """Save results for applying `rule` onto `row` into `rule_results`.loc[`i`, `rule_index`]."""
    profile = {attr: [] if attr_spec_[attr].multi_value else None for attr in attr_columns}
    for attr, value in row.items():
        if attr in attr_columns and value is not None:
            profile[attr] = convert_dataset_item(value, attr_spec_[attr])
    result = rule(profile)
    rule_results.loc[i, rule_index] = raw_to_generic_topclass(list(result.items())) if result is not None else None
    return rule_results.loc[i, rule_index] is not None


def get_confusion(rule_results, reference):
    """Get the confusion of `rule_results` compared to `reference`."""
    rule_confusion_df = pd.DataFrame(index=generic_labels, columns=rule_results.columns)
    for col in rule_results:
        if not any(rule_results[col].notna()):
            rule_confusion_df[col] = pd.Series(data=[1 for _ in generic_labels], index=generic_labels)

        cut_result = rule_results[rule_results[col].notna()][col]
        cut_reference = reference[rule_results[col].notna()]

        cm = confusion_matrix(cut_reference, cut_result, labels=generic_labels)
        cm_df = pd.DataFrame(cm, index=generic_labels, columns=generic_labels)
        for cm_col in cm_df:
            if any(cm_df[cm_col] != 0):
                rule_confusion_df[col] = cm_df[cm_col]
                break
    return rule_confusion_df


def remove_unused_rules(rule_results):
    """Drop not - NaN columns of rule results DataFrame."""
    to_drop = []
    for col in rule_results.columns:
        if not any(rule_results[col].notna()):
            to_drop.append(col)
    return rule_results.drop(columns=to_drop)


def get_normalized_confusion(all_rules, in_df, reference, attr_spec_, attr_columns):
    """Get normalized confusion df for `all_rules`."""
    # test rules individially
    print("Testing rules.", file=sys.stderr)
    rule_results = pd.DataFrame(index=in_df.index, columns=range(len(all_rules)))
    for rule_index, rule in enumerate(all_rules):
        for i, row in in_df.iterrows():
            test_rule(i, row, rule, rule_index, rule_results, attr_spec_, attr_columns)

    # remove unused rules
    print("Removing unused rules.", file=sys.stderr)
    rule_results = remove_unused_rules(rule_results)

    # Get confusion of assignment for each rule
    print("Assessing confusion.", file=sys.stderr)
    rule_confusion_df = get_confusion(rule_results, reference)

    # Normalize confusion
    print("Normalizing confusion.", file=sys.stderr)
    return rule_confusion_df / rule_confusion_df.sum()


def train_rules(train_method, mod, in_df, reference, attr_columns, discard_untrained=False):
    """Perform individual rule training based on rule confusion statistics."""
    # load all tested rules
    all_rules = load_all_rules(config)

    normalized_confusion_df = get_normalized_confusion(all_rules, in_df, reference, attr_spec, attr_columns)

    # Replace currently assigned mass functions with ones based on confusion distribution
    print("Applying changes to rules.", file=sys.stderr)
    training_rules = deepcopy(all_rules)
    all_classes = all_rules[0].all_classes

    for rule_index in normalized_confusion_df:
        if isinstance(training_rules[rule_index], OfflineRangeRule):
            continue
        training_rules[rule_index].maf = train_method(normalized_confusion_df[rule_index], all_classes, mod)

    if discard_untrained:
        training_rules = [r for i, r in enumerate(training_rules) if i in normalized_confusion_df]

    return training_rules


# Save trained rules
def export_rules(trained: list[AbstractBaseRule], filename, header=""):
    """Save `trained` rules to `filename` file."""
    rule_buffer = defaultdict(list)
    for trained_rule in trained:
        exported = trained_rule.export(taxonomy)
        class_name, rule = exported.split("\n", maxsplit=1)
        rule_buffer[class_name].append(rule)

    with open(filename, "w") as outfile:
        if header:
            outfile.write(f"#{header}\n")
        for class_name, buffered_rules in sorted(((cn, br) for cn, br in rule_buffer.items()), key=lambda x: x[0]):
            rules_string = "\n".join(sorted(buffered_rules, reverse=True))
            print(f"{class_name}\n" f"{rules_string}\n", file=outfile)


parser = deepcopy(data_io.parser)
parser.add_argument(
    "--method",
    choices=["distribute-confusion", "two-focal", "two-focal-opt"],
    default="two-focal-opt",
    help="method to transform from confusion matrix column " "to mass function (default: %(default)s)",
)
parser.add_argument(
    "--opt_function",
    choices=["f1", "f2"],
    default="f2",
    help="translation function to use for two-focal-opt option " "(default: %(default)s)",
)
parser.add_argument("-a", type=float, default=None, help="used as opt_function(a, b) (best if not entered)")
parser.add_argument("-b", type=float, default=None, help="used as opt_function(a, b) (best if not entered)")
parser.add_argument("--discard-untrained", action="store_true", help="Discard rules that remained unused in training.")
parser.add_argument("--detect-best", action="store_true", help="Detect best possible params. Ignores other options.")

method_select = {
    "distribute-confusion": get_replacement_mf,
    "two-focal": get_replacement_mf_single_class,
    "two-focal-opt": get_replacement_mf_single_class_optimized,
}

function_select = {"f1": linear_step, "f2": midpoint_transition}


def identity(param):
    return param


if __name__ == "__main__":
    args = parser.parse_args()

    config = data_io.get_config(args.config)
    module_config = data_io.get_config_section(config, args)

    ATTR_COLUMNS = data_io.get_config_item(config, "ATTR_COLUMNS")

    if not args.detect_best:
        method = method_select[args.method]
        f = function_select[args.opt_function](args.a, args.b)
    else:
        params_metrics = pd.read_csv(data_io.get_config_item(module_config, "PARAMS_METRICS"), index_col=0)
        best_opt, best_score = params_metrics["f1"].idxmax(), params_metrics["f1"].max()

        ds_metrics_paths = data_io.get_config_item(module_config, "DS_METRICS")
        ds_metrics = pd.read_csv(ds_metrics_paths["DS1"], index_col=0).rename(
            columns={"DS_teorie": "distribute-confusion"}
        )
        ds2 = pd.read_csv(ds_metrics_paths["DS2"], index_col=0)
        ds_metrics.insert(1, "two-focal", ds2["DS_teorie"])
        ds_metrics = ds_metrics[["distribute-confusion", "two-focal"]].T
        best_ds, best_ds_score = ds_metrics["f1"].idxmax(), ds_metrics["f1"].max()
        if best_ds_score > best_score:
            method = method_select[best_ds]
            f = identity
        else:
            method = method_select["two-focal-opt"]
            args.opt_function, params_str = str(best_opt).split("(", maxsplit=1)
            args.a, args.b = (float(s[:-1]) for s in params_str.split(" "))
            f = function_select[args.opt_function](args.a, args.b)

    # Load data
    attr_spec = data_io.load_attr_spec(config)
    input_df = data_io.load_raw_data(module_config, "RAW_INPUT_SOURCE")
    joined_reference = data_io.load_reference(module_config, "REFERENCE_SOURCE", input_df)

    print(
        f"Training method {method.__name__}, optimizing function {args.opt_function} "
        f"({function_select[args.opt_function].__name__}) with args a={args.a}, b={args.b}"
    )
    print(f"Using input of length {len(input_df)}.", file=sys.stderr)
    trained_rules = train_rules(
        method, f, input_df, joined_reference, ATTR_COLUMNS, discard_untrained=args.discard_untrained
    )

    export_rules(
        trained_rules,
        data_io.get_config_item(module_config, "RULE_OUTPUT"),
        header=f"{method.__name__}, optimized using {args.opt_function}, params ({args.a}, {args.b})",
    )
    print("Trained rules have been exported.", file=sys.stderr)
