"""
Module for handling the data-point data model.

Data model:
- Datapoints are aggregated by entity ID (e.g. IP address) to profiles
- Each profile is a dict, whose keys are attribute names, values are lists of tuples (t1,t2,c,val) (or 2D numpy array
  with equivalent content), "t1" and "t2" are datetime or pandas.Timestamp objects (which are almost the same thing),
  "c" is confidence (float), "val" is attribute value, it's data type is different for each attribute, it should match
  the one in loaded attribute specification.
  Example:
    {
      "open_ports": [
        (Timestamp('2021-06-18 02:17:14.855000'), Timestamp('2021-06-18 02:38:40.062000'), 1.0, list([53])),
        (Timestamp('2021-06-18 02:38:40.049000'), Timestamp('2021-06-18 03:06:00.817000'), 1.0, list([771, 53]))
      ],
      "hardware_type_ua": [
        (Timestamp('2021-06-18 03:01:06.024000'), Timestamp('2021-06-18 03:16:08.685000'), 1.0, 'computer')
      ]
    }
- Profiles are usually stored in a dict (entity_id -> profile).
"""

import argparse
import gzip
import logging
import os
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from dp3.common.attrconvert import get_converter
from dp3.common.attrspec import AttrSpec

from dependencies import dp3config

logging.basicConfig(level=logging.INFO, format="%(name)s [%(levelname)s] %(message)s")
log = logging.getLogger("DataPointLoader")
# log.setLevel("DEBUG")


# Store loaded attribute specification as global variable
# (it's not the best solution, but it simplifies things a lot)
ATTR_SPEC: Optional[dict] = None


######################################################################################################################


class DataPointLoader:
    """Loader of datapoint files as written by DP3 API receiver."""

    # Names of columns in datapoint files
    COL_NAMES = ["type", "id", "attr", "t1", "t2", "c", "src", "val"]

    def __init__(self, attr_config_dirname: str):
        """
        Create a datapoint loader.

        attr_config_dirname: Directory with attribute configuration (same as for DP3)
        """
        # Load attribute config
        attr_spec = dp3config.load_and_extend_attr_spec(attr_config_dirname)

        # Prepare a table for data type conversion
        # (to get data type from attr_spec: attr_spec[etype]["attribs"][attrname].data_type)
        self.dt_conv = {}  # map (etype,attr_name) -> conversion_function
        for etype, spec in attr_spec.items():
            for aname, aspec in spec["attribs"].items():
                self.dt_conv[(etype, aname)] = get_converter(aspec.data_type)

        self.ATTR_SPEC = attr_spec
        global ATTR_SPEC  # pylint: disable=global-statement
        ATTR_SPEC = attr_spec

    def read_dp_file(self, filename: str) -> pd.DataFrame:
        """
        Read a file with ADiCT/DP3 datapoints into pandas DataFrame.

        Values of attributes in datapoints are validated and converted according to the attribute configuration passed
        to DataPointLoader constructor.
        """
        if filename.endswith(".gz"):
            open_function = gzip.open
        else:
            open_function = open  # type: ignore

        # Reformat datapoints file so "val" containing commas can be read properly.
        #   Replace first 7 commas (i.e. all except those inside "val") with semicolon
        #   Store as "tmp" file
        tmp_name = f"tmp-{'.'.join(os.path.basename(os.path.normpath(filename)).split(sep='.')[:-1])}"
        with open_function(filename, "rb") as infile:
            with open(tmp_name, "wb") as outfile:
                for line in infile:
                    outfile.write(line.replace(b",", b";", 7))
        # Load the converted file
        data = pd.read_csv(
            tmp_name,
            sep=";",
            header=None,
            names=self.COL_NAMES,
            index_col=False,
            converters={"c": float, "val": str},
            parse_dates=["t1", "t2"],
            infer_datetime_format=True,
        )
        # Cleanup
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

        # Convert values to correct types according to attr_spec
        def convert_row(row):
            try:
                row[2] = self.dt_conv[(row[0], row[1])](row[2])
            except KeyError as e:
                raise KeyError(f"No converter for {(row[0], row[1])}, with value {row[2]}.") from e
            except ValueError:
                print("ValueError in conversion, v:", row)
                return row
            return row

        attrs = {entity_attr[1] for entity_attr in self.dt_conv}
        conv_vals = data.loc[data["attr"].isin(attrs), ("type", "attr", "val")].apply(convert_row, axis=1, raw=True)
        if len(conv_vals) != len(data):
            log.warning("Dropped %s rows due to missing attributes in config", len(data) - len(conv_vals))
            log.info("Missing attrs: %s", [x for x in data["attr"].unique() if x not in attrs])
        data["val"] = conv_vals["val"]
        return data[data["attr"].apply(lambda x: x in attrs)]


######################################################################################################################


def datapoints_to_profiles(dps: pd.DataFrame, debug=False) -> dict:
    """
    Convert a list of datapoints to profiles of individual entities.

    Datapoints should be filtered to contain only one type of entity.
    """
    if dps.empty:
        print("WARNING: Empty DataFrame, no profile generated.")
        return {}

    etype = dps["type"].unique()
    assert len(etype) == 1, (
        f"ERROR: multiple entity types found ({etype})." f" You must pass a DataFrame with only one entity type."
    )
    etype = etype[0]

    # Get list of attributes and entity IDs
    attrs = dps["attr"].unique()
    attrs.sort()
    eids = dps["id"].unique()
    eids.sort()

    # Groups for history processing (each id-attr separately)
    grouped_by_id_attr = dps.groupby(by=["id", "attr"], sort=False)
    # Select only needed columns
    grouped_by_id_attr = grouped_by_id_attr[["t1", "t2", "c", "val"]]  # FutureError

    # TODO aggregate datapoints in each group
    agg_groups = grouped_by_id_attr

    # Generate profiles as dicts of (attribute -> list_of_timestamped_values)
    profiles = {}
    for eid in eids:
        profile = {}
        for attr in attrs:
            try:
                tmp_arr = agg_groups.get_group((eid, attr)).to_numpy(dtype=object)
                unique_arr = []
                for row in tmp_arr:
                    if list(row) not in unique_arr:
                        unique_arr.append(list(row))
                profile[attr] = np.array(unique_arr, dtype=object)
            except KeyError:
                pass  # no datapoint for this combination of ip and attr
        profiles[eid] = profile
    if debug:
        print(f"Found {len(eids)} entities of type '{etype}'. Generating profiles...")
        print("Attributes:", ", ".join(attrs))
        print(f"{len(profiles)} '{etype}' profiles sucessfully generated.")
    return profiles


######################################################################################################################
# Functions to pretty-print entity profiles


def _print_timedelta(td):
    """Print timedelta as number of hours, minutes and seconds (format: '[[[#d]#h]#m]#s')"""
    res = ""
    sec = td.total_seconds()
    d, h, m, s = int(sec // 86400), int(sec % 86400 // 3600), int((sec % 3600) // 60), int(sec % 60)
    if d:
        res += f"{d}d"
    if h or d:
        res += f"{h}h"
    if m or h or d:
        res += f"{m}m"
    res += f"{s}s"
    return res


# Pretty-print IP profiles
def _print_profile(eid: str, p: dict, shorten: bool = False, attrs: set = None) -> None:
    def print_row(row):
        conf = f"(c={row[2] * 100:.0f}%) " if row[2] != 1.0 else ""
        print(
            f"        {row[0].strftime('%Y-%m-%d %H:%M:%S')} - {row[1].strftime('%Y-%m-%d %H:%M:%S')} "
            f"({_print_timedelta(row[1] - row[0])}) {conf}{row[3]}"
        )

    print(f"{eid}:")
    for attr, vals in p.items():
        if attrs is not None and attr not in attrs:
            continue
        print(f"    {attr}:")
        # simple determination if this attribute has history
        if isinstance(vals, np.ndarray):  # array of timestamped values
            if shorten and len(vals) > 10:  # shorten too long histories by skipping some rows
                for row in vals[:4]:
                    print_row(row)
                print(f"        ... ({len(vals) - 8} values skipped) ...")
                for row in vals[-4:]:
                    print_row(row)
            else:  # print all rows
                for row in vals:
                    print_row(row)
        else:  # single value
            print(f"        {vals}")


def print_profiles(profiles: dict, *eids: str, shorten: bool = False, attrs: set = None) -> None:
    """Pretty prints passed `profiles`. If `eids` are specified, print only those profiles."""
    if eids:
        # Print profiles with passed IDs
        for eid in eids:
            _print_profile(eid, profiles.get(eid, {}), shorten=shorten, attrs=attrs)
    else:
        # Print all profiles
        for eid, p in profiles.items():
            _print_profile(eid, p, shorten=shorten, attrs=attrs)


######################################################################################################################
# IP-MAC Mapping


def get_ip_to_mac_mapping(ip_mac_profiles):
    """
    Get dict with mapping of IP to list of MAC addresses

    Return dict (IP address -> [list of MAC addresses])
    """
    mapping = defaultdict(list)
    for eid in ip_mac_profiles:
        mac, ip = eid.split("-")
        mapping[ip].append(mac)
    return mapping


# Join MAC data into IP profiles
def join_mac_into_ip(ip_prof, mac_prof, ip_mac):
    """
    Add info from MAC profiles to matching IP profiles.

    For each IP, get all attributes of all MAC addresses associated with it and add them to IP's attributes.
    Also add "mac" attribute containing the list of MAC addresses.
    IP profiles are updated in place.
    """
    mapping = get_ip_to_mac_mapping(ip_mac)
    for ip, ip_attrs in ip_prof.items():
        macs = mapping.get(ip, [])
        ip_attrs["mac"] = macs
        if not macs:
            continue  # no MAC known for this IP
        for mac in macs:
            ip_attrs.update(mac_prof.get(mac, {}))


######################################################################################################################
# Helper functions


def get_attr_spec(attr_name: str) -> AttrSpec:
    """Get AttrSpec that matches an attribute name, without specifying the entity."""
    assert ATTR_SPEC is not None
    for entity_spec in ATTR_SPEC.values():
        if attr_name in entity_spec["attribs"]:
            return entity_spec["attribs"][attr_name]
    raise ValueError(f"No such attribute {attr_name}")


######################################################################################################################


def main():
    """Main module code"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="datapointloader.py", description="Module for downloading and preprocessing data for JUPYTER notebook."
    )
    parser.add_argument(
        "-s",
        "--source",
        metavar="FILENAME",
        dest="src_file",
        default=None,
        help="Path to file which should be processed.",
    )
    parser.add_argument("-c", "--config", metavar="ATTRIBUTECONFIG", dest="config", default=None, help="")

    attr = parser.parse_args()

    if attr.src_file is None:
        log.error("No log file to download was entered!")
        sys.exit(1)

    if attr.config is None:
        log.error("Configuration file for dp entities was not entered!")
        sys.exit(1)

    log.info("Loading attribute configuration from %s", attr.config)
    dpl = DataPointLoader(attr.config)
    log.info("Loading datapoint file %s", attr.src_file)
    dps = dpl.read_dp_file(attr.src_file)
    log.info("%s datapoints loaded.", len(dps))

    log.info("Generating profiles")

    # ip_profiles = datapoints_to_profiles(dps[dps["type"] == "ip"])
    # mac_profiles = datapoints_to_profiles(dps[dps["type"] == "mac"])
    # ip_mac_profiles = datapoints_to_profiles(dps[dps["type"] == "ip_mac"])


if __name__ == "__main__":
    main()
