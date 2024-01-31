"""
Module implementing K-fold cross validation testing of rule optimization.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
# pylint: disable=comparison-with-callable
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from multiprocessing import Pool, current_process
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import data_io
from basic_eval import apply_rules, convert_dataset_item, raw_to_generic_topclass
from constants import max_process_cnt
from label_fusion_classification import merge_taxonomy_classes
from pyds.pyds import MassFunction
from rule_training import (
    function_select,
    generic_labels,
    get_confusion,
    get_replacement_mf,
    get_replacement_mf_single_class,
    load_all_rules,
    method_select,
    parser,
    remove_unused_rules,
    taxonomy,
    test_rule,
)
from ruleloader.rules import AbstractRangeRule

generic_labels = sorted(generic_labels)


def train_rules(train_index, train_method, mod, in_df, reference, attr_spec_, config_):
    """Train individual rules using their confusion distribution."""
    all_rules = load_all_rules(config_)
    attr_columns_ = data_io.get_config_item(config_, "ATTR_COLUMNS")
    trained_rules = deepcopy(all_rules)

    # test rules individially
    rule_results = pd.DataFrame(index=in_df.index, columns=range(len(trained_rules)))
    for rule_index, rule in enumerate(trained_rules):
        for i in train_index:
            row = in_df.iloc[i]
            test_rule(i, row, rule, rule_index, rule_results, attr_spec_, attr_columns_)

    # remove unused rules
    rule_results = remove_unused_rules(rule_results)

    # Get confusion of assignment for each rule
    rule_confusion_df = get_confusion(rule_results, reference)

    # Normalize confusion
    normalized_confusion_df = rule_confusion_df / rule_confusion_df.sum()

    # Replace currently assigned mass functions with ones based on confusion distribution
    all_classes = all_rules[0].all_classes

    for rule_index in normalized_confusion_df:
        if isinstance(trained_rules[rule_index], AbstractRangeRule):
            continue
        trained_rules[rule_index].maf = train_method(normalized_confusion_df[rule_index], all_classes, mod)

    return trained_rules


def train_rules_sharp(train_index, train_method, mod, in_df, reference, attr_spec_, config_):
    """Train groups of rules that are activated together."""
    all_rules = load_all_rules(config_)
    attr_columns_ = data_io.get_config_item(config_, "ATTR_COLUMNS")
    trained_rules = deepcopy(all_rules)

    # Test rules individually
    rule_results = pd.DataFrame(index=in_df.index, columns=range(len(trained_rules)))
    rule_combinations = set()
    for i in train_index:
        rule_combination = []
        for rule_index, rule in enumerate(trained_rules):
            row = in_df.iloc[i]
            if test_rule(i, row, rule, rule_index, rule_results, attr_spec_, attr_columns_):
                rule_combination.append(rule_index)

        if rule_combination:
            rule_combinations.add(tuple(sorted(rule_combination)))

    # Remove unused rules
    to_drop = []
    for col in rule_results.columns:
        if not any(rule_results[col].notna()):
            to_drop.append(col)
    rule_results = rule_results.drop(columns=to_drop)

    # Get confusion for groups of rules that activated together
    trained_rule_groups = {}
    for rule_combination in rule_combinations:
        rule_group = []
        index_selector = rule_results[list(rule_combination)].notna().all(axis=1)
        rule_group_results = rule_results.loc[index_selector, list(rule_combination)]

        rule_confusion_df = get_confusion(rule_group_results, reference[index_selector])

        # Normalize confusion
        normalized_confusion_df = rule_confusion_df / rule_confusion_df.sum()

        # Replace currently assigned mass functions with ones based on confusion distribution
        all_classes = all_rules[0].all_classes
        for rule_index in normalized_confusion_df:
            rule = deepcopy(trained_rules[rule_index])
            if not isinstance(trained_rules[rule_index], AbstractRangeRule):
                rule.maf = train_method(normalized_confusion_df[rule_index], all_classes, mod)
            rule_group.append(rule)

        trained_rule_groups[rule_combination] = rule_group

    return trained_rule_groups


def encode_result(result: Optional[MassFunction]) -> list[list[float]]:
    """Encode passed MaF into vector of weights. If None is passed, weights are uniformly distributed."""
    if result is None:
        return [[1 / len(generic_labels) for _ in generic_labels]]
    class_mass = [(tuple(cls)[0], mass) for cls, mass in result.items()]
    return [[y[1] for y in sorted(class_mass, key=lambda x: x[0])]]


def simplify_rule(tax, rule):
    rule.maf = merge_taxonomy_classes(tax, rule.maf)
    return rule


def complicate_rule(tax, rule):
    """inverse of `simplify_rule()`."""
    detailed_evaluation = MassFunction()
    for classification_hypothesis, belief in rule.maf.items():
        classification = list(classification_hypothesis)[0]  # Unfreeze the frozenset
        children = tax.get_children(classification)
        detailed_evaluation[frozenset(children)] += belief
    rule.maf = detailed_evaluation.normalize()
    return rule


def strip_rule_taxonomy(rule):
    """Remove the taxonomy layer, keep only strings."""
    stripped = {}
    for hypothesis, bel in rule.maf.items():
        stripped[tuple(str(x) for x in hypothesis)] = bel
    rule.maf = MassFunction(stripped)
    return rule


def return_rule_taxonomy(rule, tax):
    """Return the taxonomy layer, assuming only string hypotheses."""
    with_taxonomy = {}
    for hypothesis, bel in rule.maf.items():
        with_taxonomy[tuple(tax.enum(x) for x in hypothesis)] = bel
    rule.maf = MassFunction(with_taxonomy)
    return rule


def clamp(val: float, lower_bound: float = 0.0001, upper_bound: float = 1.0) -> float:
    return max(lower_bound, min(val, upper_bound))


def train_rules_gradient(train_index, mod, in_df, reference, attr_spec_, config_):
    """Train rules using gradient descent."""
    all_rules = load_all_rules(config_)
    trained_rules = deepcopy(all_rules)
    trained_rules = [simplify_rule(taxonomy, r) for r in trained_rules if not isinstance(r, AbstractRangeRule)]
    trained_rules = [strip_rule_taxonomy(r) for r in trained_rules]

    le = preprocessing.LabelBinarizer()
    le.fit(generic_labels)

    rule_references = {}
    for row_index in train_index:
        profile = get_profile_from_row(row_index, in_df, attr_spec_, attr_columns)
        applied_rules = []
        for rule in trained_rules:
            res = rule(profile)
            if res is not None:
                applied_rules.append(rule)
        rule_references[row_index] = applied_rules

    prev_loss = None
    curr_loss = 0
    epsilon = 10e-3
    for _ in range(100):
        for row_index in train_index:
            optimized_rules = rule_references[row_index]
            if len(optimized_rules) == 0:
                continue

            results = [rule.maf for rule in optimized_rules]
            result = reduce(lambda a, b: a & b, results)

            # Gradient descent
            mass_step = 0.001
            learning_constant = 0.001
            optimized_mfs = [r.maf for r in optimized_rules]

            encoded_reference = le.transform([reference[row_index]])
            encoded_result = encode_result(result)
            base_loss = mean_squared_error(encoded_reference, encoded_result)
            curr_loss += base_loss
            # for every rule
            for i, maf in enumerate(optimized_mfs):
                new_mf = {}
                if len(results) == 1:
                    static_results = None
                else:
                    static_results = reduce(lambda a, b: a & b, results[:i] + results[i + 1 :])
                en_static_results = np.array(encode_result(static_results)[0])
                en_mass_values = np.array([mass for label, mass in sorted(maf.items(), key=lambda x: x[0])])

                # for every class / mass value
                for label_i, label in enumerate(generic_labels):
                    base_mass = maf[(label,)]

                    en_mass_values[label_i] = clamp(base_mass + mass_step)
                    skip = en_mass_values[label_i] - base_mass == 0

                    en_result = en_static_results * en_mass_values
                    en_mass_values[label_i] = base_mass
                    en_result /= en_result.sum()
                    new_loss = mean_squared_error(encoded_reference[0], en_result)

                    if skip:
                        new_mf[(label,)] = base_mass
                        continue

                    # Get "engineering" derivative
                    loss_derivative = (new_loss - base_loss) / abs(mass_step)

                    # new mass = old mass - learning constant * (derivative of loss / derivative of mass) (clamped)
                    # as derivative of mass is always 1, we can simplify the formula to:
                    new_mf[(label,)] = clamp(base_mass - learning_constant * loss_derivative)

                optimized_rules[i].maf = MassFunction(new_mf).normalize()

        if prev_loss is not None and abs(curr_loss - prev_loss) <= epsilon:
            break

        print(current_process().name, curr_loss)
        prev_loss = curr_loss
        curr_loss = 0

    trained_rules = [return_rule_taxonomy(r, taxonomy) for r in trained_rules]
    trained_rules = [complicate_rule(taxonomy, r) for r in trained_rules]
    for rule in all_rules:
        if isinstance(rule, AbstractRangeRule):
            trained_rules.append(rule)

    return trained_rules


def get_metrics(cut_results, cut_reference):
    """Return metrics for given results."""

    precision_res = precision_score(cut_reference, cut_results, average="weighted", zero_division=0)
    recall_res = recall_score(cut_reference, cut_results, average="weighted", zero_division=0)
    f1_res = f1_score(cut_reference, cut_results, average="weighted", zero_division=0)
    accuracy_res = accuracy_score(cut_reference, cut_results)
    return precision_res, recall_res, f1_res, accuracy_res


fallback_count = 0


def get_sharp_rules_result(profile: dict, rules: dict, test_results: pd.Series, i: int):
    """Sharp rules have different "classifier" structure."""
    detected_group = []
    for rule_index, rule in enumerate(rules[None]):
        if rule(profile) is not None:
            detected_group.append(rule_index)
    detected_group_tuple = tuple(sorted(detected_group))
    if detected_group_tuple in rules:
        aggregate_rule_results(profile, rules[detected_group_tuple], test_results, i)
    else:  # Fallback to trained rules
        global fallback_count  # pylint: disable=global-statement
        fallback_count += 1
        aggregate_rule_results(profile, rules[None], test_results, i)


def aggregate_rule_results(profile: dict, rules: list, test_results: pd.Series, i: int):
    """Apply rules, aggregate results, return generic topclass."""
    result = apply_rules(rules, profile)
    if result is not None:
        test_results.loc[i] = raw_to_generic_topclass(list(result.items())) if result is not None else None


def test_rules(
    rules, test_index, in_df, reference, attr_spec_, attr_columns_, classification_function=aggregate_rule_results
):
    """Test passed rules against the given reference."""
    test_results = pd.Series(index=in_df.index, dtype=object)

    for i in test_index:
        profile = get_profile_from_row(i, in_df, attr_spec_, attr_columns_)
        classification_function(profile, rules, test_results, i)

    prec, recall, f1, acc = get_metrics(test_results[test_results.notna()], reference[test_results.notna()])
    print(f"{prec:15.3f}|{recall:15.3f}|{f1:15.3f}|{acc:15.3f}")
    return prec, recall, f1, acc, test_results


def test_rules_priority(rules, test_index, in_df, reference, attr_columns_):
    """Test passed rules using the priority method."""
    all_classes = rules[0].all_classes

    def get_priority(rule):
        if isinstance(rule, AbstractRangeRule):
            return rule.base_mass + rule.eased_mass
        return max(c for h, c in rule.maf.items() if tuple(h) != all_classes)

    test_results = pd.Series(index=in_df.index, dtype=object)
    sorted_rules = sorted(deepcopy(rules), key=get_priority, reverse=True)

    for i in test_index:
        profile = get_profile_from_row(i, in_df, attr_spec, attr_columns_)
        result = None
        for rule in sorted_rules:
            result = rule(profile)
            if result is not None:
                break

        if result is not None:
            test_results.loc[i] = raw_to_generic_topclass(list(result.items())) if result is not None else None

    p, r, f1, a = get_metrics(test_results[test_results.notna()], reference[test_results.notna()])
    return p, r, f1, a, test_results


def test_rules_majority(rules, test_index, in_df, reference, attr_columns_):
    """Test passed rules using the weighted majority voting method."""
    all_classes = rules[0].all_classes

    def get_weight(rule):
        if isinstance(rule, AbstractRangeRule):
            return rule.base_mass + rule.eased_mass
        return max(c for h, c in rule.maf.items() if tuple(h) != all_classes)

    test_results = pd.Series(index=in_df.index, dtype=object)
    weights = [get_weight(r) for r in rules]

    for i in test_index:
        profile = get_profile_from_row(i, in_df, attr_spec, attr_columns_)
        voting_weights = defaultdict(float)
        for rule, weight in zip(rules, weights):
            result = rule(profile)
            if result is not None:
                voting_weights[raw_to_generic_topclass(list(result.items()))] += weight

        if len(voting_weights) == 0:
            test_results.loc[i] = None
        else:
            test_results.loc[i] = sorted(list(voting_weights.items()), key=lambda x: x[1], reverse=True)[0][0]

    p, r, f1, a = get_metrics(test_results[test_results.notna()], reference[test_results.notna()])
    return p, r, f1, a, test_results


def get_profile_from_row(i, in_df, attr_spec_, attr_columns_):
    """Return a device profile based on row defined by `i` and `in_df`."""
    row = in_df.iloc[i]
    profile = {attr: [] if attr_spec_[attr].multi_value else None for attr in attr_columns_}
    for attr, attr_value in row.items():
        if attr in attr_columns_ and attr_value is not None:
            profile[attr] = convert_dataset_item(attr_value, attr_spec_[attr])
    return profile


def test_rules_oracle(rules, test_index, in_df, reference, attr_columns_):
    """Test the oracle method."""
    test_results = pd.Series(index=in_df.index, dtype=object)

    for i in test_index:
        profile = get_profile_from_row(i, in_df, attr_spec, attr_columns_)
        result = None
        for rule in rules:
            tmp = rule(profile)
            if tmp is not None:
                result = raw_to_generic_topclass(list(tmp.items()))
                if result == joined_reference.loc[i]:
                    break

        test_results.loc[i] = result

    p, r, f1, a = get_metrics(test_results[test_results.notna()], reference[test_results.notna()])
    return p, r, f1, a, test_results


def test_classifier(classifier, train_index, test_index, in_df, reference, attr_spec_, config_):
    """Train and test given classifier."""
    rules = load_all_rules(config_)
    attr_columns_ = data_io.get_config_item(config_, "ATTR_COLUMNS")

    rule_results = pd.DataFrame(index=in_df.index, columns=range(len(rules)))
    for rule_index, rule in enumerate(rules):
        for i, row in in_df.iterrows():
            test_rule(i, row, rule, rule_index, rule_results, attr_spec_, attr_columns_)
    rule_results = rule_results.replace([None], 0)
    rule_results = rule_results.replace(r"OperatingSystem.*", 1, regex=True)

    train_index_selector = [i in train_index for i in input_df.index]
    test_index_selector = [i in test_index for i in input_df.index]

    classifier.fit(rule_results[train_index_selector], reference[train_index_selector])
    test_results = classifier.predict(rule_results[test_index_selector])

    p, r, f1, a = get_metrics(test_results, reference[test_index_selector])
    return p, r, f1, a, pd.Series(test_results)


sorted_to_run_choices = ["priority", "majority", "oracle", "dt", "nb", "rf", "ab"]
parser.add_argument(
    "--also-run",
    nargs="+",
    choices=sorted_to_run_choices,
    help="priority - perform priority merging with trained rules; "
    "majority - perform weighted majority voting with trained rules; "
    "oracle - perform oracle merging with trained rules; "
    "dt - use decision tree classifier; "
    "rf - use random forest classifier; "
    "ab - use ADA boost classifier; ",
    default=None,
)
parser.add_argument("-k", default=5, dest="k", type=int, help="the K in K-fold validation (default: %(default)s)")
parser.add_argument("--out", default="", type=str, help="suffix for the output file")
parser.add_argument(
    "--distortion-resilience",
    action="store_true",
    help="whether to perform testing of resilience (requires special config)",
)


def update_measured(
    measured_values: list[dict], method_index: int, i: int, p: float, r: float, f1: float, a: float, res: pd.Series
):
    """Update measured values."""
    measured_values[method_index][(i, "precision")] = p
    measured_values[method_index][(i, "recall")] = r
    measured_values[method_index][(i, "f1")] = f1
    measured_values[method_index][(i, "accuracy")] = a
    measured_values[method_index][(i, "results")] = res


def test_indexes(i_split_indicies, method_, fn, in_df, reference, attr_spec_, attr_columns_, config_, to_run: list):
    """Test given indices using the provided `method`."""
    i, split_indicies = i_split_indicies
    train_index, test_index = split_indicies
    measured_metrics: list[dict[tuple[int, str], Any]] = [{} for _ in range(2 + len(to_run))]

    trained_rules = train_rules(train_index, method_, fn, in_df, reference, attr_spec_, config_)
    metrics = test_rules(trained_rules, test_index, in_df, reference, attr_spec_, attr_columns_)
    update_measured(measured_metrics, 0, i, *metrics)

    if method_ == get_replacement_mf:
        trained_rules_sharp = train_rules_sharp(train_index, method_, fn, in_df, reference, attr_spec_, config_)
        trained_rules_sharp[None] = trained_rules
        metrics = test_rules(
            trained_rules_sharp,
            test_index,
            in_df,
            reference,
            attr_spec_,
            attr_columns_,
            classification_function=get_sharp_rules_result,
        )
        print(f"Fallback count of sharp rules: {fallback_count} of {len(test_index)}")
        update_measured(measured_metrics, 1, i, *metrics)

    elif method_ == get_replacement_mf_single_class:
        trained_rules_gradient = train_rules_gradient(train_index, fn, in_df, reference, attr_spec_, config_)
        metrics = test_rules(trained_rules_gradient, test_index, in_df, reference, attr_spec_, attr_columns_)
        update_measured(measured_metrics, 1, i, *metrics)

    if "priority" in to_run:
        metrics = test_rules_priority(trained_rules, test_index, in_df, reference, attr_columns_)
        update_measured(measured_metrics, 2 + to_run.index("priority"), i, *metrics)

    if "majority" in to_run:
        metrics = test_rules_majority(trained_rules, test_index, in_df, reference, attr_columns_)
        update_measured(measured_metrics, 2 + to_run.index("majority"), i, *metrics)

    if "oracle" in to_run:
        metrics = test_rules_oracle(trained_rules, test_index, in_df, reference, attr_columns_)
        update_measured(measured_metrics, 2 + to_run.index("oracle"), i, *metrics)

    classifiers = {
        "dt": DecisionTreeClassifier(),
        "rf": RandomForestClassifier(),
        "ab": AdaBoostClassifier(),
    }
    for classifier_name, classifier in classifiers.items():
        if classifier_name in to_run:
            metrics = test_classifier(classifier, train_index, test_index, in_df, reference, attr_spec_, config_)
            print(classifier_name, metrics[:4])
            update_measured(measured_metrics, 2 + to_run.index(classifier_name), i, *metrics)

    return measured_metrics


def setup_distortion_resilience(module_config_, args_k, random_state=42):
    """
    Set up the testing of distortion resilience.
    Different data sources / distributions for training and testing are assumed.
    """
    train_df = data_io.load_raw_data(module_config_, "TRAIN_SOURCE")
    train_reference = data_io.load_reference(module_config_, "TRAIN_REFERENCE_SOURCE", train_df)

    test_df = data_io.load_raw_data(module_config_, "TEST_SOURCE")
    test_reference = data_io.load_reference(module_config_, "TEST_REFERENCE_SOURCE", test_df)

    kf = KFold(n_splits=args_k, shuffle=True, random_state=random_state)

    train_splits = [x[0] for x in kf.split(train_df)]
    test_splits = [x[1] + len(train_df) for x in kf.split(test_df)]
    indexes = zip(train_splits, test_splits)

    in_df = train_df.append(test_df, ignore_index=True)
    reference = train_reference.append(test_reference, ignore_index=True)
    return in_df, reference, indexes


if __name__ == "__main__":
    args = parser.parse_args()
    if args.also_run is None:
        args.to_run = []
    else:
        args.to_run = [x for x in sorted_to_run_choices if x in args.also_run]
    print(args.to_run)

    method = method_select[args.method]
    f = function_select[args.opt_function](args.a, args.b)

    config = data_io.get_config(args.config)
    module_config = data_io.get_config_section(config, args)

    # Load configuration
    RANDOM_STATE = data_io.get_config_item(config, "RANDOM_STATE")
    METRICS_OUT = data_io.get_config_item(module_config, "METRICS_OUT")
    RESULTS_OUT = data_io.get_config_item(module_config, "RESULTS_OUT")

    # Load data
    attr_spec = data_io.load_attr_spec(config)
    attr_columns = data_io.get_config_item(config, "ATTR_COLUMNS")

    k = args.k
    if args.distortion_resilience:
        input_df, joined_reference, split_indexes = setup_distortion_resilience(module_config, k, RANDOM_STATE)
    else:
        input_df = data_io.load_raw_data(module_config, "RAW_INPUT_SOURCE")
        joined_reference = data_io.load_reference(module_config, "REFERENCE_SOURCE", input_df)

        kf5 = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        split_indexes = kf5.split(input_df)

    df_outs = {
        name: pd.DataFrame(index=range(k), columns=["precision", "recall", "f1", "accuracy"])
        for name in [args.out, f"{args.out}#", *args.to_run]
    }

    def testing_function(x):
        return test_indexes(
            x,
            method,
            f,
            input_df,
            joined_reference,
            attr_spec,
            attr_columns,
            config,
            args.to_run,
        )

    print(f"{'Precision':15s}|{'Recall':15s}|{'F1 score':15s}|{'Accuracy':15s}")
    print(f"{15 * '-'}|{15 * '-'}|{15 * '-'}|{15 * '-'}")

    # Main processing loop
    with Pool(min(max_process_cnt // 2, k)) as pool:
        data_out = pool.map(testing_function, enumerate(split_indexes))

    results_df = pd.DataFrame(columns=[args.out, f"{args.out}#", *args.to_run], index=input_df.index)
    for item in data_out:
        for measured, target_name in zip(item, df_outs):
            for i_metric, value in measured.items():
                index, metric = i_metric
                if metric == "results":
                    results_df[target_name].fillna(value, inplace=True)
                    continue
                df_outs[target_name].loc[index, metric] = value

    if args.out:
        split = METRICS_OUT.split(".")
        METRICS_OUT = f"{split[0]}-{args.out}.{'.'.join(split[1:])}"
        split = RESULTS_OUT.split(".")
        RESULTS_OUT = f"{split[0]}_{args.out}.{'.'.join(split[1:])}"

    results_df.rename(
        columns={
            "priority": "PV+",
            "majority": "WMV",
            "oracle": "ORA+",
            "dt": "DT",
            "nb": "NB",
            "rf": "RF",
            "ab": "AB",
        }
    ).to_csv(RESULTS_OUT)

    pd.DataFrame({colname: df.mean() for colname, df in df_outs.items()}).rename(
        columns={
            args.out: "DS_teorie",
            f"{args.out}#": "DS#_teorie",
            "priority": "Prioritni",
            "majority": "Vetsinove",
            "oracle": "Oracle",
            "dt": "DT",
            "nb": "NB",
            "rf": "RF",
            "ab": "AB",
        }
    ).to_csv(METRICS_OUT)
