"""
Implementation an interface for the DP3 platform.
author: Ondřej Sedláček <ondrej.sedlacek@cesnet.cz>
"""
import logging
from typing import Any, Callable

from dp3.common.attrconvert import get_converter, get_element_type, is_iterable


class DP3Ifc:
    """Interface to the DP3 platform."""

    def __init__(self, attr_spec):
        self.attr_spec = attr_spec

        # Prepare a table for data type conversion
        # (to get data type from attr_spec: attr_spec[etype]["attribs"][attrname].data_type)
        self.dt_conv = {}  # map (attr_name) -> conversion_function
        self.iterable_attrs = set()
        self.have_confidence = set()
        for spec in self.attr_spec.values():
            for aname, aspec in spec["attribs"].items():
                if aname in self.dt_conv:
                    if self.dt_conv[aname] != get_converter(aspec.data_type):
                        logging.error(
                            "Duplicit attribute names with different convertors between entities - %s: %s vs %s",
                            aname,
                            self.dt_conv[aname],
                            get_converter(aspec.data_type),
                        )

                if is_iterable(aspec.data_type):
                    self.iterable_attrs.add(aname)
                    element_type = get_element_type(aspec.data_type)

                    if isinstance(element_type, dict):
                        self._typenames_dict_to_convertor_dict(element_type)
                        self.dt_conv[aname] = element_type
                        continue

                    self.dt_conv[aname] = get_converter(element_type)
                else:
                    self.dt_conv[aname] = get_converter(aspec.data_type)

                if aspec.multi_value:
                    self.iterable_attrs.add(aname)

                if aspec.confidence:
                    self.have_confidence.add(aname)

    @staticmethod
    def _typenames_dict_to_convertor_dict(type_dict: dict[str, Callable[[str], Any]]) -> None:
        for attr_name, attr_type in type_dict.items():
            type_dict[attr_name] = get_converter(attr_type)

    def get_attr_converter(self, attr_name):
        return self.dt_conv[attr_name]

    def get_dict_attr_key_converter(self, attr_name, key):
        return self.dt_conv[attr_name][key]

    def is_iterable(self, attr_name) -> bool:
        return attr_name in self.iterable_attrs

    def has_confidence(self, attr_name) -> bool:
        return attr_name in self.have_confidence
