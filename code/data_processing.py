import pandas as pd
import numpy as np


def generate_ship_lengths(
    sample_size,
    data_csv="data/Total Accumalted Transits by Market Segment and Lock FY 2024.csv",
    length_ranges_csv="data/Length Ranges.csv",
):
    """
    Generate a list of ship lengths representing the underlying dataset.

    Parameters:
    - sample_size (int): The total number of ships to sample.
    - data_csv (str): Path to the CSV file containing ship data.
    - length_ranges_csv (str): Path to the CSV file containing length ranges.

    Returns:
    - ship_lengths_df (DataFrame): DataFrame containing the sampled ship lengths.
    """

    # Load the Dataset
    df = pd.read_csv(data_csv)
    length_ranges_df = pd.read_csv(length_ranges_csv)
    ship_lengths = []  # Empty list to store ship lengths

    # Calculate Total Ships and Proportions
    df["Total"] = df["NeoPanamax"] + df["Panamax"]
    total_ships = df["Total"].sum()
    df["Proportion"] = df["Total"] / total_ships

    # Determine Sample Sizes for Each Ship Type
    df["Sample Size"] = (df["Proportion"] * sample_size).round().astype(int)

    # Adjust Sample Size if Total Doesn't Sum to Desired Sample Size
    difference = sample_size - df["Sample Size"].sum()
    if difference != 0:
        # Adjust the sample size for the ship type with the largest proportion
        max_idx = df["Proportion"].idxmax()
        df.loc[max_idx, "Sample Size"] += difference

    for idx, row in df.iterrows():
        ship_type = row["Ship Type"]
        total_ships_type = row["Total"]
        neo_ships = row["NeoPanamax"]
        panamax_ships = row["Panamax"]
        sample_size_type = row["Sample Size"]

        if sample_size_type == 0:
            continue  # Skip if no ships are to be sampled for this type

        # Calculate the proportion of NeoPanamax and Panamax ships within this type
        neo_ratio = neo_ships / total_ships_type if total_ships_type > 0 else 0
        panamax_ratio = panamax_ships / total_ships_type if total_ships_type > 0 else 0

        # Determine how many ships of each canal category to sample
        neo_sample_size = int(round(sample_size_type * neo_ratio))
        panamax_sample_size = (
            sample_size_type - neo_sample_size
        )  # Ensure total matches the sample size

        # Generate ship lengths for NeoPanamax ships
        if neo_sample_size > 0:
            # Get length ranges for this ship type and NeoPanamax category
            range_row = length_ranges_df[
                (length_ranges_df["Ship Type"] == ship_type)
                & (length_ranges_df["Canal"] == "NeoPanamax")
            ]
            if not range_row.empty:
                low = range_row["Min Length"].values[0]
                high = range_row["Max Length"].values[0]
                lengths = np.random.uniform(low, high, neo_sample_size)
                for length in lengths:
                    ship_lengths.append(
                        {
                            "Ship Type": ship_type,
                            "Canal": "NeoPanamax",
                            "Length (m)": round(length, 2),
                        }
                    )

        # Generate ship lengths for Panamax ships
        if panamax_sample_size > 0:
            # Get length ranges for this ship type and Panamax category
            range_row = length_ranges_df[
                (length_ranges_df["Ship Type"] == ship_type)
                & (length_ranges_df["Canal"] == "Panamax")
            ]
            if not range_row.empty:
                low = range_row["Min Length"].values[0]
                high = range_row["Max Length"].values[0]
                lengths = np.random.uniform(low, high, panamax_sample_size)
                for length in lengths:
                    ship_lengths.append(
                        {
                            "Ship Type": ship_type,
                            "Canal": "Panamax",
                            "Length (m)": round(length, 2),
                        }
                    )

    # Compile the List into a DataFrame
    ship_lengths_df = pd.DataFrame(ship_lengths)

    # Reset the index
    ship_lengths_df.reset_index(drop=True, inplace=True)

    # Return the final DataFrame
    return ship_lengths_df
