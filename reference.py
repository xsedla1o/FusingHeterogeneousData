"""
Module implementing aligning the reference data to the dataset.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""

import pandas as pd

import data_io


def translate_shodan_reference(shodan_os):
    """Translate shodan values to our taxonomy."""
    if not shodan_os:
        return ""
    if "Windows" in shodan_os:
        return "OperatingSystem.Windows"
    if shodan_os in ["Debian", "Ubuntu", "Raspbian"]:
        return "OperatingSystem.Linux"
    raise ValueError(f"Unknown shodan OS value: {shodan_os}")


def fill_reference(reference_ids, reference_values: dict):
    """Fill reference values using annotation sources."""
    priority = ["http_ua", "shodan"]
    reference_df = pd.DataFrame({"id": reference_ids}, columns=["id", "results"]).fillna("")
    for source in priority:
        if source not in reference_values:
            continue
        source_df = reference_values[source]
        merged_df = pd.DataFrame({"id": reference_ids, "results": reference_df["results"], "ref_source": source_df["ref_source"]}).fillna("")

        reference_df["results"] = merged_df["results"].where(merged_df["results"] != "", merged_df["ref_source"])
        print(reference_df.results.value_counts())
    return reference_df.results


parser = data_io.parser
args = parser.parse_args()

config = data_io.get_config(args.config)
module_config = data_io.get_config_section(config, args)

# Load configuration
REFERENCE_OUT = data_io.get_config_item(module_config, "REFERENCE_OUT")

# Load data
input_df = data_io.load_raw_data(module_config, "RAW_INPUT_SOURCE")
result_df_1 = data_io.load_results(module_config, "LABEL_RESULTS_SOURCE")

reference_sources = {
    "http_ua": pd.DataFrame({"id": input_df.index, "ref_source": result_df_1["http_ua"]}).fillna("").astype(str),
    "shodan":  pd.DataFrame({"id": input_df.index, "ref_source": input_df["shodan_os_extracted"].apply(translate_shodan_reference).astype(str)}),
}

# Join both reference sources
joined_reference = fill_reference(input_df.index, reference_sources)
joined_reference.to_csv(REFERENCE_OUT)
