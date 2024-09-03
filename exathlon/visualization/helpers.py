"""Visualization helpers module.

Gathers a set of visualization helpers common to all datasets.
"""
import importlib

from utils.common import USED_DATA, DATA_CONFIG

# set visualization configuration and functions depending on the input data
VISUALIZATION_CONFIG = importlib.import_module(
    f"utils.{USED_DATA}"
).VISUALIZATION_CONFIG
get_period_title_from_info = importlib.import_module(
    f"utils.{USED_DATA}"
).get_period_title_from_info

# expand visualization configuration with label colors and legends from anomaly type indices
VISUALIZATION_CONFIG["label_colors"] = {0: VISUALIZATION_CONFIG["normal_color"]}
VISUALIZATION_CONFIG["label_legends"] = {0: "NORMAL"}
for label in range(1, len(DATA_CONFIG["anomaly_types"]) + 1):
    VISUALIZATION_CONFIG["label_colors"][label] = VISUALIZATION_CONFIG[
        "anomaly_colors"
    ][DATA_CONFIG["anomaly_types"][label - 1]]
    VISUALIZATION_CONFIG["label_legends"][label] = DATA_CONFIG["anomaly_types"][
        label - 1
    ].upper()


class Color:
    """Color codes for displaying colors in the console."""

    def __init__(self):
        pass

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
