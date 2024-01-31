"""Module for synthetic dataset generation."""
import json
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import data_io
from constants import module_to_attrs


def get_module_os_distribution(sel_modules: list[str], os_d, module_cms):
    """Returns an OS distribution supported by selected modules."""
    possible_os = set(os_d.index)
    for sel_module in sel_modules:
        possible_os &= set(module_cms[sel_module].columns[~module_cms[sel_module].isna().all()])
    return os_d[possible_os] / os_d[possible_os].sum()


def get_os_module_distribution(annotated_os: str, module_cms, module_d):
    """Returns an OS distribution supported by selected modules."""
    possible_modules = set()
    for module_name, module_cm in module_cms.items():
        if ~module_cm[annotated_os].isna().all():
            possible_modules.add(module_name)
    possible_modules.remove("http_ua")
    return module_d[possible_modules] / module_d[possible_modules].sum()


def session_syntesis(  # pylint: disable=R0914
    overlap_d, module_d, os_d, module_cms, rng: np.random.Generator
) -> tuple[str, dict]:
    """Return: session annotation, session attribute values."""
    generated_session = {}

    # How many modules will have data?
    overlap = rng.choice(overlap_d.index, p=overlap_d["0"])

    # What will the annotated OS be?
    annotated_os = rng.choice(os_d.index, p=os_d[overlap])

    # What modules will give data?
    modules = choose_modules(annotated_os, module_cms, module_d, overlap, rng)

    # Generate classification output for each module based on its confusion matrix
    for module_name in modules:
        try:
            module_classified = rng.choice(module_cms[module_name].index, p=module_cms[module_name][annotated_os])
        except ValueError as e:
            print(e)
            print(module_name, annotated_os, module_cms[module_name][annotated_os].sum())
            raise

        # Fill all module's attributes
        for attribute in module_to_attrs[module_name]:
            class_attr_values = attr_values[attribute][module_classified]
            if len(class_attr_values["values"]) == 0:
                print(f"Skipped cause no value: {module_name}, {attribute}, {module_classified}")
                continue  # no value for this attribute
            if pd.isna(class_attr_values["mean"]) and pd.isna(class_attr_values["std"]):
                selected_attr_value = rng.choice(class_attr_values["values"], size=None, replace=False)
                generated_session[attribute] = selected_attr_value
                continue

            number_of_values = max(1, int(r.normal(class_attr_values["mean"], class_attr_values["std"])))
            number_of_values = min(len(class_attr_values["values"]), number_of_values)
            selected_attr_values = rng.choice(
                class_attr_values["values"], size=number_of_values if number_of_values != 1 else None, replace=False
            )

            selected_attr_values = attr_values_to_list(attribute, selected_attr_values)

            generated_session[attribute] = selected_attr_values
    return annotated_os, generated_session


def choose_modules(annotated_os, module_cms, module_d, overlap, rng):
    """Choose modules based on given distribution."""
    module_choices = get_os_module_distribution(annotated_os, module_cms, module_d[overlap])
    modules = list(
        rng.choice(module_choices.index, p=module_choices, size=min(len(module_choices.index), overlap), replace=False)
    )
    modules.append("http_ua")
    return modules


def attr_values_to_list(attribute, selected_attr_values):
    """Transform selected attr values to native python list."""
    if "tls_os" in attribute:  # TLS fingerprinting needs special treatment
        selected_attr_values = list(selected_attr_values)
    elif isinstance(selected_attr_values, np.ndarray):
        selected_attr_values = list(selected_attr_values)
    else:
        selected_attr_values = [selected_attr_values]
    return selected_attr_values


if __name__ == "__main__":
    parser = data_io.parser
    parser.add_argument("N", type=int, help="Size of the generated dataset in sessions.")
    parser.add_argument("--full-synth", action="store_true", help="Whether to generate imaginary modules.")
    args = parser.parse_args()

    config = data_io.get_config(args.config)
    module_config = data_io.get_config_section(config, args)

    if args.full_synth:
        module_to_attrs = {x: [x] for x in "abcdefghi"} | {"http_ua": module_to_attrs["http_ua"]}

    ATTR_COLUMNS = data_io.get_config_item(config, "ATTR_COLUMNS")

    # Load dataset characteristics
    with open(data_io.get_config_item(module_config, "ATTR_VALS")) as in_file:
        attr_values = json.load(in_file)

    # Load distributions
    overlap_distribution = pd.read_csv(data_io.get_config_item(module_config, "OVERLAP"), index_col=0)

    module_distribution = pd.read_csv(data_io.get_config_item(module_config, "MODULE"), index_col=0)
    module_distribution = module_distribution.rename(columns={c: int(c) for c in module_distribution.columns})

    # drop annotation attribute (for now at least)
    module_distribution = module_distribution.drop(index=["http_ua"])
    module_distribution /= module_distribution.sum()

    os_distribution = pd.read_csv(data_io.get_config_item(module_config, "OS"), index_col=0)
    os_distribution = os_distribution.rename(columns={c: int(c) for c in os_distribution.columns})

    # Set module confusion matricies
    module_confusion_matricies = {}
    for module, path in data_io.get_config_item(module_config, "MODULE_CM_PATHS").items():
        module_confusion_matricies[module] = pd.read_csv(path, index_col=0)

    generated_dataset_path = data_io.get_config_item(module_config, "DATASET_OUT")

    # Init random generator
    r = np.random.default_rng()

    # Disable warning triggered when generating TLS attributes
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    # Main synthesis loop
    generated_dataset = {
        attr: [None for i in range(args.N)] for attr in
        ["annot", "shodan_os_extracted"] + sorted(ATTR_COLUMNS)
    }

    for i in tqdm(range(args.N)):
        annot, session = session_syntesis(
            overlap_d=overlap_distribution,
            module_d=module_distribution,
            os_d=os_distribution,
            module_cms=module_confusion_matricies,
            rng=r,
        )
        generated_dataset["annot"][i] = annot
        for attr, val in session.items():
            generated_dataset[attr][i] = val

    dataset_df = pd.DataFrame(generated_dataset)
    dataset_df.to_csv(generated_dataset_path)
