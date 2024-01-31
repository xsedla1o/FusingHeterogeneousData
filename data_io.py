"""
Module implementing loading from configuration and data files.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
from argparse import Action, ArgumentParser, FileType

import pandas as pd
from yaml import safe_load

from dependencies.datapointloader import DataPointLoader

default_config = "config/experiments.yml"


class SplitArgs(Action):
    def __call__(self, arg_parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))


def get_config(config):
    return safe_load(config)


def get_config_section(config, arguments):
    """Get config section dedicated to module, based on arguments."""
    config_section = config
    for i, item in enumerate(arguments.config_section):
        if item not in config_section:
            raise KeyError(
                f"Missing subsection {item} in config {arguments.config.name}{arguments.config_section[:i]}."
            )
        config_section = config_section[item]
    return config_section


def get_config_item(config_section, item):
    """Return item from given config."""
    if item in config_section:
        return config_section[item]
    raise KeyError(f"Required item {item} is missing from section {config_section}.")


def load_attr_spec(config):
    """Load and return the attribute specification."""

    path = get_config_item(config, "ATTR_CONF_DIR")
    dpl = DataPointLoader(path)
    attr_spec = dpl.ATTR_SPEC

    return {**attr_spec["ip"]["attribs"], **attr_spec["mac"]["attribs"]}


def load_raw_data(config_section, data_name):
    """Load the input dataset based on config section and name."""
    input_df = pd.read_csv(get_config_item(config_section, data_name))
    return input_df.where(pd.notnull(input_df), None)


def load_results(config_section, source_name):
    """Load the results of classifying dataset based on config section and name."""
    result_df_1 = pd.read_csv(get_config_item(config_section, source_name)).drop(columns=["Unnamed: 0"])
    result_df_1 = result_df_1.where(pd.notnull(result_df_1), "")
    return result_df_1


def load_joined_results(config_section, results_source):
    """Load the merged results of classifying dataset based on config section and name."""
    return pd.read_csv(get_config_item(config_section, results_source))


def load_reference(config_section, reference_name, input_df):
    """Load the reference based on config section and name."""
    reference = pd.read_csv(get_config_item(config_section, reference_name)).drop(columns=["Unnamed: 0"]).results
    reference.index = input_df.index
    return reference


parser = ArgumentParser()
parser.add_argument("config", type=FileType("r"), help="YML configuration file containing data paths")
parser.add_argument(
    "config_section", type=str, action=SplitArgs, help="comma separated list of keys leading to module's config"
)
