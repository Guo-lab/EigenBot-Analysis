import argparse

import pandas as pd


def combine_and_sort_csv(files, output_file):
    # Initialize an empty list to hold the DataFrames
    dataframes = []

    # Loop through the list of files and read each into a DataFrame
    for file in files:
        df = pd.read_csv(f"{file}_data.csv")
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.sort_values(by="Timestamp")

    combined_df.to_csv(output_file, index=False)
    print(f"Combined and sorted data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine and sort CSV files by 'Timestamp'."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of CSV files (without extension) to combine.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file name for the combined and sorted data.",
    )

    args = parser.parse_args()
    print(args)

    # Call the function to combine and sort the CSV files
    combine_and_sort_csv(args.files, args.output)
