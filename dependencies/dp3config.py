"""Module enabling the use of DP3 based functions for internal LabelFusion tasks."""
from dp3.common.config import load_attr_spec, read_config_dir


def load_and_extend_attr_spec(attr_conf_dir):
    """Loads the attribute specification and extends it with missing attributes and rule evaluation attributes."""
    return load_attr_spec(read_config_dir(attr_conf_dir))
