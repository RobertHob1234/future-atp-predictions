"""This script is meant only to reformat and process data from Jeff Sackmann's atp database
    db : https://github.com/JeffSackmann/tennis_atp
    Robert Hoang
    2025-07-17"""

from pathlib import Path
import pandas as pd

def filter_from_database():
    raw_dir = Path(r"D:\Tennis Data\tennis_atp")
    out_dir = Path(r"D:\tennis_processed")
    out_dir.mkdir(parents = True, exist_ok = True)

    # relevant columns
    keep = [
        "surface",
        "winner_hand", "winner_age", "winner_rank", "winner_rank_points",
        "loser_hand", "loser_age", "loser_rank", "loser_rank_points"
    ]
    rows = 0
    for year in range(1990, 2025):
        src = raw_dir / f"atp_matches_{year}.csv"
        if not src.exists():
            print(f"Skipping missing file for {year}")
            continue

        df = pd.read_csv(src, usecols = keep)

        # only keep complete pre-match rows
        df = df.dropna(subset = keep)
        dst = out_dir / f"cleaned_{year}.csv"
        df.to_csv(dst, index = False)
        rows += len(df)
        print(f" {year}: {len(df)} rows -> {dst.name}")

    print(f"We processed {rows} rows. Save this number")

# puts all training data into one csv for one example matrix X
def combine_all_files():
    # folder containing your per-year cleaned files
    processed_dir = Path(r"D:\tennis_processed")

    # find and sort all the yearly files
    files = sorted(processed_dir.glob("cleaned_*.csv"))

    # read them all into a list, then concatenate
    dfs = [pd.read_csv(f) for f in files]
    all_matches = pd.concat(dfs, ignore_index=True)

    # write out one master csv file
    out_path = processed_dir / "all_matches_1990_2024.csv"
    all_matches.to_csv(out_path, index=False)

    print(f"Combined {len(dfs)} files → {len(all_matches)} total rows")

# randomly shuffles data into around 214,922 examples X.shape = (9,214922)
def reformat_and_shuffle():
    # read csv
    cols = [
        "surface",
        "winner_hand", "winner_age", "winner_rank", "winner_rank_points",
        "loser_hand", "loser_age", "loser_rank", "loser_rank_points"
    ]
    df = pd.read_csv(r"D:\tennis_processed\all_matches_1990_2024.csv", usecols=cols)

    # winner to player_0 loser to player_1 for randomness
    df0 = df.rename(columns={
        "winner_hand": "player_0_hand",
        "winner_age": "player_0_age",
        "winner_rank": "player_0_rank",
        "winner_rank_points": "player_0_rank_points",
        "loser_hand": "player_1_hand",
        "loser_age": "player_1_age",
        "loser_rank": "player_1_rank",
        "loser_rank_points": "player_1_rank_points",
    })
    df0["Output"] = 1

    # swapped version
    df1 = df0.copy()
    for feat in ["hand", "age", "rank", "rank_points"]:
        a = f"player_0_{feat}"
        b = f"player_1_{feat}"
        df1[a], df1[b] = df1[b].values, df1[a].values
    df1["Output"] = 0

    # I be shufflin and shit
    full = pd.concat([df0, df1], ignore_index=True)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)

    # print to csv
    full.to_csv(r"D:\tennis_processed\dataset_for_training.csv", index=False)
    print(f"Final dataset: {full.shape[0]} rows, with {full['Output'].mean():.2%} positives")

def cleanse_data():
    df = pd.read_csv(r"D:\tennis_processed\dataset_for_training.csv")
    valid_hands = {'R', 'L'}
    before = len(df)
    # keep only rows where both hands are valid
    df_clean = df[
        df['player_0_hand'].isin(valid_hands) &
        df['player_1_hand'].isin(valid_hands)
    ].reset_index(drop=True)
    dropped = before - len(df_clean)
    print(f"Dropped {dropped} rows with invalid hand codes.")
    df_clean.to_csv(r"D:\tennis_processed\dataset_for_training.csv", index=False)


def encode_values():
    # load your dataset
    df = pd.read_csv(r"D:\tennis_processed\dataset_for_training.csv")

    # map surface to integers. Low cardinality so we're fine with basic encoding
    surface_map = {"Grass": 0, "Hard": 1, "Clay": 2, "Carpet": 3}
    df["surface_code"] = df["surface"].map(surface_map)

    # map hand to integers R to 0, L to 1
    hand_map = {"R": 0, "L": 1}
    df["p0_hand_code"] = df["player_0_hand"].map(hand_map)
    df["p1_hand_code"] = df["player_1_hand"].map(hand_map)

    # drop the original text columns
    df = df.drop(["surface", "player_0_hand", "player_1_hand"], axis=1)
    df.to_csv(r"D:\tennis_processed\dataset_encoded.csv", index=False)

    print("Saved encoded dataset to dataset_encoded.csv")

    # inspect
    print(df[["surface_code", "p0_hand_code", "p1_hand_code"]].head())

# the data was verified in excel by creating unique identifiers for each game

if __name__ == "__main__":
    #filter_from_database()
    #combine_all_files()
    #reformat_and_shuffle()
    cleanse_data()
    encode_values()