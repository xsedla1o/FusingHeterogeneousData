"""Module for extracting dataset characteristics, such as observed attribute values and OS distribution."""
import ast
import json

import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import data_io
from constants import attr_columns, attr_to_module, generic_labels, module_to_attrs


class ValuesEncoder(json.JSONEncoder):
    """JSON encoder that handles python `set` type."""

    def default(self, o):
        if isinstance(o, set):
            return list(o)
        return super().default(o)


def is_hashable(attr: str, attr_spec) -> bool:
    return "dict" not in attr_spec[attr].data_type


def proc_operating_system_ua(classification, values):
    """Process "operating_system_ua" attribute values."""
    if classification == "OperatingSystem.Windows":
        return {x for x in values if "Windows" in x}
    if classification == "OperatingSystem.Linux":
        return {x for x in values if any(y in x for y in ["Ubuntu", "Linux", "CentOS", "Fedora", "Debian", "FreeBSD"])}
    if classification == "OperatingSystem.MacOS":
        return {x for x in values if "Mac OS" in x}
    if classification == "OperatingSystem.iOS":
        return {x for x in values if "iOS" in x}
    return values


def proc_tls_os_family(classification, values):
    """Process "tls_os_family" attribute values."""
    if classification == "OperatingSystem.Windows":
        print(classification, values)
        return {x for x in values if any("Win" in y for y in x)}
    if classification == "OperatingSystem.Linux":
        return {x for x in values if any("Linux" in y for y in x)}
    return values


# Data structure configuration


# Post-processing
post_processing = {
    "os_by_tcpip": lambda y, x: x,  # No action
    "operating_system_ua": proc_operating_system_ua,  # Filtering
    "hardware_type_ua": lambda y, x: x,  # No action
    "operating_platform_ua": lambda y, x: x,  # No action
    "tags_by_services": lambda y, x: x,  # No action
    "open_ports": lambda y, x: x,  # No action
    "tls_os_family": proc_tls_os_family,  # Filtering
    "tls_os_name": lambda y, x: x,  # No action
    "tls_os_version": lambda y, x: x,  # No action
    "tls_categories": lambda y, x: x,  # No action
    "sdp_label": lambda y, x: x,  # No action
    "dnssd_service": lambda y, x: x,  # No action
    "dnssd_query": lambda y, x: x,  # No action
    "ssdp_query": lambda y, x: x,  # No action
    "ssdp_service": lambda y, x: x,  # No action
}


def extract_attr_values(config_, module_config_):
    """Extract values of attributes in the dataset."""
    # Loading data
    attr_spec = data_io.load_attr_spec(config_)
    input_df = data_io.load_raw_data(module_config_, "RAW_INPUT_SOURCE")
    result_df = data_io.load_results(module_config_, "RESULTS_SOURCE")

    attr_values_path = data_io.get_config_item(module_config_, "ATTR_VALS_OUT")

    check_all_attrs_array_compliant(attr_spec)

    # Gather all values
    attr_len, attr_values = gather_attr_len_values(attr_spec, input_df, result_df)

    # Gather mean and std of value counts
    attr_mean_std = get_len_stats(attr_len, result_df)

    # Post-processing
    sanitize_values(attr_mean_std, attr_values)

    out_obj = {
        attr: {
            c: {
                "values": sorted(attr_values[attr][c]) if is_hashable(attr, attr_spec) else attr_values[attr][c],
                "mean": attr_mean_std[attr][c]["mean"],
                "std": attr_mean_std[attr][c]["std"],
            }
            for c in generic_labels
        }
        for attr in attr_columns
    }

    # Serialize and save
    with open(attr_values_path, "w") as out_file:
        json.dump(out_obj, out_file, cls=ValuesEncoder, indent=2)


def sanitize_values(attr_mean_std, attr_values):
    """Perform post-processing to sanitize certain attributes."""
    for attr, classification_values in attr_values.items():
        print(attr)
        for classification, values in classification_values.items():
            mean_std = attr_mean_std[attr][classification]
            print(classification, mean_std["mean"], mean_std["std"])

            attr_values[attr][classification] = post_processing[attr](classification, values)
            print(classification, attr_values[attr][classification])
        print()


def check_all_attrs_array_compliant(attr_spec):
    """Check all attrs, whether any is not multi_value or array"""
    for attr, spec in attr_spec.items():
        if attr not in attr_columns:
            continue
        assert spec.multi_value or "array" in spec.data_type, f"Panic: Attribute {attr} can't be turned into a list."


def gather_attr_len_values(attr_spec, input_df, result_df):
    """Gather attr item lengths and values."""
    attr_values = {
        attr: {c: set() if is_hashable(attr, attr_spec) else [] for c in generic_labels} for attr in attr_columns
    }
    attr_len = pd.DataFrame(columns=attr_columns, index=input_df.index).fillna(0)
    for i, row in tqdm(result_df.iterrows()):
        for module, classification in row[row != ""].items():
            for attr, values in input_df.iloc[i][module_to_attrs[module]].items():
                if values is None:
                    continue
                python_value = ast.literal_eval(values)

                # TLS fingerprinting needs special treatment
                if "tls_os" in attr:
                    python_value = [tuple(python_value)]

                # Save values
                if is_hashable(attr, attr_spec):
                    attr_values[attr][classification].update(python_value)
                elif python_value not in attr_values[attr][classification]:
                    attr_values[attr][classification].extend(python_value)

                # Save length
                length = len(python_value)
                if length != 0:
                    attr_len.loc[i, attr] = length
    return attr_len, attr_values


def get_len_stats(attr_len, result_df):
    """Return the mean and std of the provided attr_len dataframe."""
    attr_mean_std = {attr: {c: {} for c in generic_labels} for attr in attr_columns}
    for attr in attr_mean_std:
        for classification in generic_labels:
            col = attr_len.loc[(attr_len[attr] != 0) & (result_df[attr_to_module[attr]] == classification), attr]
            attr_mean_std[attr][classification] = {"mean": col.mean(), "std": col.std(ddof=0)}
    return attr_mean_std


def extract_distributions(config_, module_config_):  # pylint: disable=R0914
    """Extract the OS distribution of the dataset."""
    # Load data
    module_cm_paths = data_io.get_config_item(module_config_, "MODULE_CM_PATHS")
    result_df = data_io.load_results(module_config_, "SANITIZED_RESULTS_SOURCE")
    joined_reference = data_io.load_reference(module_config_, "SANITIZED_REFERENCE_SOURCE", result_df)
    joined_results = data_io.load_joined_results(module_config_, "JOINED_RESULTS_SOURCE")["0"]

    module_distribution_path = data_io.get_config_item(module_config_, "MODULE")
    overlap_distribution_path = data_io.get_config_item(module_config_, "OVERLAP")
    os_distribution_path = data_io.get_config_item(module_config_, "OS")

    # Extracting confusion matricies
    for col, path in module_cm_paths.items():
        cut_results = result_df[result_df[col] != ""]
        reference = joined_reference[result_df[col] != ""]
        cm = confusion_matrix(reference, cut_results[col], labels=generic_labels)
        cm_df = pd.DataFrame(cm, index=generic_labels, columns=generic_labels).T
        cm_df /= cm_df.sum()
        cm_df.to_csv(path)

    # Module distribution
    active_counts = (
        (result_df["os_by_tcpip"] != "").astype(int)
        + (result_df["os_by_tls"] != "").astype(int)
        + (result_df["sdp_labels"] != "").astype(int)
        + (result_df["tags_by_services"] != "").astype(int)
    )
    module_counts = pd.DataFrame(index=result_df.columns, columns=range(1, active_counts.max() + 1), dtype=int)
    for col in module_counts.columns:
        selected = result_df[active_counts == col]
        for module in module_counts.index:
            module_counts.loc[module, col] = selected[selected[module] != ""].shape[0]

    for n in module_counts.columns:
        module_counts[n] = module_counts[n].astype("int64")

    (module_counts / module_counts.sum()).to_csv(module_distribution_path)

    print("Extracting overlap distribution")
    # Overlap distribution
    module_counts_ttl = module_counts.T.loc[:, module_counts.index != "http_ua"].T.sum()
    for index, count in module_counts_ttl.items():
        module_counts_ttl[index] = count / index

    module_counts_ttl = module_counts_ttl / module_counts_ttl.sum()

    module_counts_ttl.to_csv(overlap_distribution_path)

    # OS distribution
    os_distribution = pd.DataFrame(index=generic_labels, columns=range(1, active_counts.max() + 1), dtype=int)
    os_distribution = fill_distribution(active_counts, joined_reference, joined_results, os_distribution)
    os_distribution /= os_distribution.sum()

    os_distribution.to_csv(os_distribution_path)


def fill_distribution(active_counts, value_series, non_empty_series, distribution):
    """Fill distribution of `value_series` into prepared distribution's values."""
    for n in distribution.columns:
        counts = value_series[(active_counts == n) & (non_empty_series != "")].value_counts()
        for os_val, value in counts.items():
            distribution.loc[os_val, n] = value
    distribution = distribution.fillna(0)
    for n in distribution.columns:
        distribution[n] = distribution[n].astype("int64")
    return distribution


if __name__ == "__main__":
    args = data_io.parser.parse_args()

    config = data_io.get_config(args.config)
    module_config = data_io.get_config_section(config, args)

    extract_attr_values(config, module_config)
    extract_distributions(config, module_config)
