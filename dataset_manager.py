"""Dataset manager module. Enables having experiment results of multiple datasets and quick switching inbetween them."""
# pylint: disable=undefined-loop-variable; config len is asserted at the beginning of all functions
import argparse
import json
import os
import pathlib
import shutil
import sys

config_path = "data/config.json"

data_dir = "data"
stash_dir = "data/stash"

data_dirs = [
    "generated",
    "source",
    "summaries",
    "synthesis",
]


def load_config():
    with open(config_path) as in_file:
        config = json.load(in_file)
    return config["data"]


def save_config(config):
    with open(config_path, "w") as out_file:
        json.dump({"data": config}, out_file, indent=2)


def update_symlinks(new_path):
    """Move symlinks from `data_dir` subdirectories to now point to `new_path` subdirectories."""
    for directory in data_dirs:
        os.symlink(
            os.path.abspath(os.path.join(new_path, directory)),
            os.path.join(data_dir, f"{directory}_"),
            target_is_directory=True,
        )
        os.replace(os.path.join(data_dir, f"{directory}_"), os.path.join(data_dir, directory))


def print_basic_status(config):
    """Print current dataset and the total number of datasets."""
    active = config[0]
    path = os.path.abspath(active["path"])
    length = len(config)
    print(f"Current dataset '{active['id']}', located in '{path}'. Total datasets: {length}")


def print_status(*_):
    """Print all dataset names, their descriptions and paths. Indicate the current dataset."""
    if os.path.isfile(config_path):
        config = load_config()
    else:
        config = []

    if len(config) == 0:
        print("No datasets found, use add command to add new.")
        return

    print(f"{' ' * 7}|{'ID':20s}|{'Description':50s}|{'Path'}")
    for i, item in enumerate(config):
        print(f"{'Active>' if i == 0 else ' ' * 7} {item['id']:20s} {item['description']:50s} {item['path']}")
    print()
    print_basic_status(config)


def add_dataset(program_args):
    """Add a new dataset into the database. Creates the empty directories to be filled with real or synthetic data."""
    assert program_args.id != "", "The dataset id must be specified when adding a new dataset."

    if os.path.isfile(config_path):
        config = load_config()
    else:
        config = []
    print(f"Loaded config file of {len(config)} items.")

    for item in config:
        assert program_args.id != item["id"], "Dataset id must be unique, choose a different one."

    if len(config) > 0:
        active_config = config[0]
        active_config["path"] = os.path.join(stash_dir, active_config["id"])

    init_data_folder(os.path.join(stash_dir, program_args.id))
    update_symlinks(os.path.join(stash_dir, program_args.id))

    config.insert(
        0,
        {
            "id": program_args.id,
            "description": program_args.description,
            "path": os.path.join(stash_dir, program_args.id),
        },
    )
    print(f'Added a new dataset entry with id "{program_args.id}", description "{program_args.description}".')
    save_config(config)
    print_basic_status(config)


def switch_dataset(program_args):
    """
    Switch from currently active dataset to another in the database.
    If no parameter is passed, barrel shift switching is performed.
    """
    assert os.path.isfile(config_path), "Configuration file is needed for switching."
    config = load_config()

    assert len(config) > 1, "Must have more than one dataset to switch between"
    print(f"Loaded config file of {len(config)} items.")

    if program_args.id == "":
        print("Barrel shift switching.")
        active = config.pop(0)
        new = config[0]
    else:
        print(f"Switching to dataset with id {program_args.id}.")
        active = config.pop(0)
        assert active["id"] != program_args.id, "Selected dataset is already active."
        for i, item in enumerate(config):
            if item["id"] == program_args.id:
                break
        else:
            raise AssertionError("Specified ID not found.")
        new = config.pop(i)
        config.insert(0, new)

    update_symlinks(os.path.join(stash_dir, new["id"]))

    config.append(active)
    save_config(config)
    print_basic_status(config)


def remove_dataset(program_args):
    """Remove the dataset based on passed ID."""
    assert program_args.id != "", "The dataset id must be specified when removing a dataset."
    assert os.path.isfile(config_path), "Configuration file is needed for removing."

    config = load_config()

    assert len(config) >= 1, "No datasets to remove."
    print(f"Loaded config file of {len(config)} items.")

    print(f"Removing dataset with id {program_args.id}.")
    for i, item in enumerate(config):
        if item["id"] == program_args.id:
            break
    else:
        raise AssertionError("Specified ID not found.")

    removed = config.pop(i)

    if i == 0:
        active = config[0]
        update_symlinks(active["path"])

    shutil.rmtree(os.path.abspath(removed["path"]))

    save_config(config)
    print_basic_status(config)


def init_data_folder(new_path):
    """Initialize the data folders for a new dataset."""
    for path in [
        "generated/",
        "source/datapoints/",
        "summaries/",
        "synthesis/",
    ]:
        pathlib.Path(os.path.join(new_path, path)).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        type=str,
        choices=["status", "add", "switch", "remove"],
        help="what should happen with currently saved datasets",
    )
    parser.add_argument("id", type=str, help="id of the selected dataset", default="", nargs="?")
    parser.add_argument("description", type=str, help="description when adding a new dataset", default="", nargs="?")
    args = parser.parse_args()

    pathlib.Path(stash_dir).mkdir(parents=True, exist_ok=True)
    try:
        if args.action == "add":
            add_dataset(args)
        elif args.action == "switch":
            switch_dataset(args)
        elif args.action == "remove":
            remove_dataset(args)
        elif args.action == "status":
            print_status(args)
    except AssertionError as err:
        print(err)
        sys.exit(1)
