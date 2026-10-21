"""Laboratory 01 Home Assignment: Standard Dataset Inspection Utility."""

import os
import sys

# Ensure lab01/inspect.py does not shadow Python's built-in 'inspect' module when numpy/pandas import it
_script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _script_dir:
    sys.path.append(sys.path.pop(0))

import pandas as pd


def inspect(path_or_url):
    """Load a CSV and print a compact, standard first-look report.

    Parameters
    ----------
    path_or_url : str
        A local file path or an http(s) URL pointing at a CSV file.
    """
    df = pd.read_csv(path_or_url)

    print("=" * 70)
    print("SOURCE :", path_or_url)
    print("SHAPE  :", df.shape[0], "rows x", df.shape[1], "columns")
    print("=" * 70)

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("None.")
    else:
        report = pd.DataFrame({
            "missing": miss,
            "percent": (miss / len(df) * 100).round(1),
        })
        print(report.sort_values("missing", ascending=False))

    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe())

    return df


if __name__ == "__main__":
    datasets = sys.argv[1:] or [
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv",
    ]
    for ds in datasets:
        inspect(ds)
        print()
