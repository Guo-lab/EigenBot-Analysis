import argparse

import pandas as pd


def update_timestamp_column(file_path, add_value):
    df = pd.read_csv(file_path)

    if "Timestamp" in df.columns:
        df["Timestamp"] = df["Timestamp"] + add_value

        df.to_csv(file_path, index=False)
        print(f"Updated {file_path} successfully.")
    else:
        print(f"Error: 'Timestamp' column not found in {file_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update 'Timestamp' column in a CSV file."
    )
    parser.add_argument("--file", type=str, required=True, help="Path to the CSV file.")
    parser.add_argument(
        "--add_value",
        type=float,
        required=True,
        help="Value to add to the 'Timestamp' column.",
    )

    args = parser.parse_args()
    update_timestamp_column(args.file, args.add_value)
