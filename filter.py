"""
Module implementing filtration of data based on specified criteria.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
import pandas as pd

import data_io

RANDOM_STATE = 1


def count_concurrent_sources(in_df: pd.DataFrame) -> pd.Series:
    """Return the number of concurrent data sources in label result data."""
    concurrent_sum = pd.Series(0, index=in_df.index, dtype=int)
    for col in in_df.columns:
        if col == "http_ua":
            continue
        concurrent_sum += (in_df[col] != "").astype(int)
    return concurrent_sum


if __name__ == "__main__":
    parser = data_io.parser
    parser.description = "Filter data based on module overlap in sessions."
    parser.add_argument("n", metavar="N", type=int, help="number of concurrent data sources required")
    parser.add_argument(
        "-s", "--strinct", dest="strict", action="store_true", help="use == instead of >= when filtering"
    )
    args = parser.parse_args()

    config = data_io.get_config(args.config)
    module_config = data_io.get_config_section(config, args)

    # Load config
    RAW_INPUT_OUT = data_io.get_config_item(module_config, "RAW_INPUT_OUT")
    LABEL_RESULTS_OUT = data_io.get_config_item(module_config, "LABEL_RESULTS_OUT")
    REFERENCE_OUT = data_io.get_config_item(module_config, "REFERENCE_OUT")

    # Load data
    input_df = data_io.load_raw_data(module_config, "RAW_INPUT_SOURCE")
    result_df_1 = data_io.load_results(module_config, "LABEL_RESULTS_SOURCE")
    joined_reference = data_io.load_reference(module_config, "REFERENCE_SOURCE", input_df)

    concurrent = count_concurrent_sources(result_df_1)
    if args.strict:
        selected = concurrent == args.n
    else:
        selected = concurrent >= args.n

    input_df = input_df[selected].reset_index().drop(columns=["index"])
    result_df_1 = result_df_1[selected].reset_index().drop(columns=["index"])
    joined_reference = joined_reference[selected].reset_index().drop(columns=["index"])

    input_df.to_csv(RAW_INPUT_OUT)
    result_df_1.to_csv(LABEL_RESULTS_OUT)
    joined_reference.to_csv(REFERENCE_OUT)
