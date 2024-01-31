"""
Module implementing dataset sanitization.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
import pandas as pd

import data_io

parser = data_io.parser
parser.description = "Sanitize and optionally balance the dataset."
parser.add_argument(
    "-b", "--balance", dest="balance", action="store_true", help="balance the dataset (Windows and Linux classes)"
)
args = parser.parse_args()

config = data_io.get_config(args.config)
module_config = data_io.get_config_section(config, args)

# Load configuration
RANDOM_STATE = data_io.get_config_item(config, "RANDOM_STATE")

RAW_INPUT_OUT = data_io.get_config_item(module_config, "RAW_INPUT_OUT")
LABEL_RESULTS_OUT = data_io.get_config_item(module_config, "LABEL_RESULTS_OUT")
JOINED_RESULTS_OUT = data_io.get_config_item(module_config, "JOINED_RESULTS_OUT")
REFERENCE_OUT = data_io.get_config_item(module_config, "REFERENCE_OUT")

# Load data
attr_spec = data_io.load_attr_spec(config)
input_df = data_io.load_raw_data(module_config, "RAW_INPUT_SOURCE")
result_df_1 = data_io.load_results(module_config, "LABEL_RESULTS_SOURCE")
joined_generic = data_io.load_joined_results(module_config, "JOINED_RESULTS_SOURCE")
joined_reference = data_io.load_reference(module_config, "REFERENCE_SOURCE", input_df)


if args.balance:
    # Sanitize and balance the dataset
    selector = pd.Series(False, index=input_df.index, dtype=bool)
    for col in result_df_1.columns:
        if col == "http_ua":
            continue
        selector |= result_df_1[col] != ""
    usable_reference = joined_reference[selector]

    usable_reference = usable_reference.dropna()
    class_distribution = usable_reference.value_counts()
    win_lin_limit = min(class_distribution["OperatingSystem.Linux"], class_distribution["OperatingSystem.Windows"])

    win = usable_reference[usable_reference == "OperatingSystem.Windows"].sample(
        win_lin_limit, random_state=RANDOM_STATE
    )
    lin = usable_reference[usable_reference == "OperatingSystem.Linux"].sample(win_lin_limit, random_state=RANDOM_STATE)
    other = usable_reference[
        (usable_reference != "OperatingSystem.Linux") & (usable_reference != "OperatingSystem.Windows")
    ]
    balanced_reference = pd.concat([win, lin, other]).sort_index()
    selected = pd.Series(index=input_df.index, dtype=bool, data=[i in balanced_reference.index for i in input_df.index])

    input_df = input_df[selected].reset_index().drop(columns=["index"])
    result_df_1 = result_df_1[selected].reset_index().drop(columns=["index"])
    joined_generic = joined_generic[selected].reset_index()["0"]
    joined_reference = balanced_reference.reset_index().drop(columns=["index"])
else:
    # Sanitize the dataset (drop where reference is NaN)
    selected = ~joined_reference.isna()
    input_df = input_df[selected].reset_index().drop(columns=["index"])
    result_df_1 = result_df_1[selected].reset_index().drop(columns=["index"])
    joined_generic = joined_generic[selected].reset_index()["0"]
    joined_reference = joined_reference[selected].reset_index().drop(columns=["index"])

input_df.to_csv(RAW_INPUT_OUT)
result_df_1.to_csv(LABEL_RESULTS_OUT)
joined_generic.to_csv(JOINED_RESULTS_OUT)
joined_reference.to_csv(REFERENCE_OUT)
