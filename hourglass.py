""" Simple supervisor script to get reports of multiple synthetic datasets. """
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time
from distutils.dir_util import copy_tree, remove_tree

from yaml import safe_load

in_dir = "config/hourglass_in"
out_dir = "config/hourglass_out"
current_path = "config/synthesis/current"
current_data = "data/generated"

# Source: https://www.asciiart.eu/miscellaneous/hourglass
hourglass = """ ______________________
(0RGSDOFCJftli;:.:. .  )
 T\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"T
 |.;....,..........;..|
 |;;:: .  .    .      |
 l;;;:. :   .     ..  ;
 `;;:::.: .    .     .'
  l;;:. ..  .     .: ;
  `;;::.. .    .  ; .'
   l;;:: .  .    /  ;
    \\;;:. .   .,'  /
     `\\;:.. ..'  .'
       `\\;:.. ..'
         \\;:. /
          l; f
          `;f'
           ||
           ;l.
          ;: l
         / ;  \\
       ,/  :   `.
     ./' . :     `.
    /' ,'  :       \\
   f  /  . :        i
  ,' ;  .  :        `.
  f ;  .   :      .  i
 .'    :   :       . `.
 f ,  .    ;       :  i
 |    :  ,/`.       : |
 |    ;,/;:. `.     . |
 |___,/;;:. . .`._____|
(QB0ZDOLC7itz!;:.:. .  )
 \"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\""""

pocket_watch = """              .+.
          _.-//_\\\\-._
        .'.-' XII '-.'.
      /`.'*         *'.`\\
     / /*        /    *\\ \\
    | ;        _/       ; |
    | |IX     (_)    III| |
    | ;         \\       ; |
     \\ \\*        \\    */ /
      \\ '.*       \\ *.'./
       '._'-.__VI_.-'_.'
          '-.,___,.-'"""


class ColumnPrinter:
    """Class enabling printing of separate left column alongside the usual arguments."""

    def __init__(self, lwidth: int, lcolumn: str, **kwargs):
        self.lwidth = lwidth
        self.lcolumn = iter(x for x in lcolumn.split("\n"))
        self.kwargs = kwargs

    def print(self, *arguments: str, **kwargs):
        """Built-in print function wrapper, but also prints left column text."""
        kwargs = {**self.kwargs, **kwargs}
        right = " ".join(arguments).split(sep="\n")
        for line in right:
            left = next(self.lcolumn, "")
            pad = max(self.lwidth - len(left), 0)
            print(left, " " * pad, line, **kwargs)

    def finalize(self):
        for remaining in self.lcolumn:
            print(remaining, **self.kwargs)


parser = argparse.ArgumentParser()
parser.add_argument("--full-synth", action="store_true", help="Whether to generate imaginary modules.")
args = parser.parse_args()

pathlib.Path(in_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

to_process = os.listdir(in_dir)
to_process_full = [os.path.join(in_dir, x) for x in to_process]

dirs_to_process = [path for path, full_path in zip(to_process, to_process_full) if os.path.isdir(full_path)]
combinations_to_process = [
    path
    for path, full_path in zip(to_process, to_process_full)
    if os.path.isfile(full_path) and (full_path.endswith(".yml") or full_path.endswith(".yaml"))
]

if len(to_process) == 0:
    sys.exit()

print(
    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
    f"Found {len(to_process)} directories to process, processing."
)

for directory in dirs_to_process:
    log = ColumnPrinter(30, hourglass)
    log.print(("\n" * 5) + f"Processing '{directory}'\n")

    start = time.time()
    loc_start = time.localtime()
    log.print(f"Start - {time.strftime('%Y-%m-%d %H:%M:%S', loc_start)}\n\n")

    # Make symlink to current
    os.symlink(os.path.abspath(os.path.join(in_dir, directory)), f"{current_path}_", target_is_directory=True)
    os.replace(f"{current_path}_", current_path)

    # Make dataset
    log.print("Generating dataset")
    if args.full_synth:
        process_result = subprocess.run("make fully_synthetic_dataset", shell=True, capture_output=True, check=False)
    else:
        process_result = subprocess.run("make synthetic_dataset_current", shell=True, capture_output=True, check=False)

    # Prepare output dir
    out_path = pathlib.Path(os.path.join(out_dir, directory))
    out_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_path, "synth_log.txt"), "w") as out:
        out.write("STDOUT\n")
        out.write(process_result.stdout.decode("utf-8"))
        out.write("STDERR\n")
        out.write(process_result.stderr.decode("utf-8"))

    # If process failed, abort.
    if process_result.returncode != 0:
        log.print("Failed, aborting")
        log.finalize()
        continue
    log.print("Generated successfully")
    end = time.time()
    log.print(f"{end - start:.2f}s elapsed\n\n")

    # Make report
    log.print("Generating report")
    process_result = subprocess.run("make report -j 8", shell=True, capture_output=True, check=False)

    out_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_path, "report_log.txt"), "w") as out:
        out.write("STDOUT\n")
        out.write(process_result.stdout.decode("utf-8"))
        out.write("STDERR\n")
        out.write(process_result.stderr.decode("utf-8"))

    # If process failed, abort.
    if process_result.returncode != 0:
        log.print("Failed, aborting")
        log.finalize()
        continue
    log.print("Generated successfully")
    mid, end = end, time.time()
    log.print(f"{end - mid:.2f}s elapsed\n\n")

    # Move processed in dir to out dir
    log.print("Moving processed directory")
    copy_tree(os.path.join(in_dir, directory), os.path.join(out_path))
    remove_tree(os.path.join(in_dir, directory))

    # Save report
    log.print("Saving report")
    shutil.copy("latex/report.pdf", os.path.join(out_dir, f"{directory}.pdf"))
    copy_tree("latex/generated", os.path.join(out_dir, f"{directory}-generated-plots"))
    copy_tree("data/generated", os.path.join(out_dir, f"{directory}-generated-data"))

    end = time.time()
    loc_end = time.localtime()
    log.print(f"\nFinished in total of {end - start:.2f}s\n")
    log.print(f"End - {time.strftime('%Y-%m-%d %H:%M:%S', loc_end)}")
    log.finalize()

for config_name in combinations_to_process:
    log = ColumnPrinter(30, pocket_watch)
    log.print(f"\nProcessing '{config_name}'\n")

    loc_start = time.localtime()
    log.print(f"Start - {time.strftime('%Y-%m-%d %H:%M:%S', loc_start)}\n")

    # Read config - train set, test set
    with open(os.path.join(in_dir, config_name)) as config_in:
        config = safe_load(config_in)
    train_path = os.path.join(out_dir, f"{config['train_source']}-generated-data")
    test_path = os.path.join(out_dir, f"{config['test_source']}-generated-data")

    # Test required files exist
    required_files = [
        required_path
        for path in [train_path, test_path]
        for required_path in [
            os.path.join(path, "sanitized_input.csv"),
            os.path.join(path, "sanitized_reference.csv"),
        ]
    ]
    missing_files = [file_path for file_path in required_files if not os.path.isfile(file_path)]
    if missing_files:
        log.print(f"Required files {missing_files} do not exist, aborting.")
        log.finalize()
        continue

    # Clean before beginning
    subprocess.run("make clean", shell=True, capture_output=True, check=False)

    # Make simlinks to given paths
    current_files = [
        os.path.join(current_data, x)
        for x in [
            "train_input.csv",
            "train_reference.csv",
            "test_input.csv",
            "test_reference.csv",
        ]
    ]
    for source, destination in zip(required_files, current_files):
        if os.path.exists(destination):
            os.remove(destination)
        os.symlink(os.path.abspath(source), destination)

    # Run target
    log.print("Running distortion resilience test.")
    process_result = subprocess.run(
        "make test_distortion_resilience -j 3", shell=True, capture_output=True, check=False
    )

    # Save results
    out_path = pathlib.Path(os.path.join(out_dir, config["dir"]))
    out_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_path, "log.txt"), "w") as out:
        out.write("STDOUT\n")
        out.write(process_result.stdout.decode("utf-8"))
        out.write("STDERR\n")
        out.write(process_result.stderr.decode("utf-8"))

    # If process failed, abort.
    if process_result.returncode != 0:
        log.print("Failed, aborting")
        log.finalize()
        continue
    log.print("Success.\n")

    # Move processed config
    log.print("Moving processed config and saving results.\n")
    shutil.copy(os.path.join(in_dir, config_name), os.path.join(out_path))
    os.remove(os.path.join(in_dir, config_name))

    # Save results
    for result_name in [
        "kfold_dr_results-ds1.csv",
        "kfold_dr_results-ds2.csv",
    ]:
        shutil.copy(os.path.join("data/generated", result_name), os.path.join(out_path, result_name))

    loc_end = time.localtime()
    log.print(f"End  -  {time.strftime('%Y-%m-%d %H:%M:%S', loc_end)}")
    log.finalize()

    # Cleanup
    for destination in current_files:
        if os.path.exists(destination):
            os.remove(destination)
