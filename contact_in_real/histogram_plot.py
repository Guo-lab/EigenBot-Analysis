import argparse
import itertools
import json
import test

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from entropy import plot_period_length_based_histogram
from scipy.stats import entropy
from tqdm import tqdm

parser = argparse.ArgumentParser(description="as the file label")
parser.add_argument(
    "--leg_number",
    type=int,
    default=17,
    help="The leg number to calculate the entropy and histogram",
)
parser.add_argument(
    "--group",
    type=str,
    default="",
    help="The group of the data to calculate the entropy and histogram",
)
args = parser.parse_args()

json_file_path = f"{args.group}_{args.leg_number}_time_span_series.json"

try:
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
        new_010_time_span_length_series = data["new_010_time_span_length_series"]
        new_01_time_span_length_series = data["new_01_time_span_length_series"]
        new_10_time_span_length_series = data["new_10_time_span_length_series"]
        plot_period_length_based_histogram(
            args,
            new_010_time_span_length_series,
            new_01_time_span_length_series,
            new_10_time_span_length_series,
        )

except FileNotFoundError:
    print("File not found")
