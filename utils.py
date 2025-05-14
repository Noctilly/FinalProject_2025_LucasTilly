import pandas as pd
import numpy as np


def compute_averages(df: pd.Dataframe):
    # Copy Dataframe and fix time format
    data_avg_per_user = df.copy()
    data_avg_per_user["time"] = pd.to_datetime(data_avg_per_user["time"])

    # Get hour from time column
    data_avg_per_user["hour"] = data_avg_per_user["time"].dt.hour

    # Compute averages of all numeric values for each user
    data_avg_per_user = (
        data_avg_per_user.groupby("user_id").mean(numeric_only=True).reset_index()
    )

    # Drop columns which don't make sense as averages
    data_avg_per_user = data_avg_per_user.drop(columns=["video_id", "like"])

    # Rename columns for clarity
    data_avg_per_user = data_avg_per_user.rename(
        columns={"watch_ratio": "watch_ratio_mean", "video_length": "video_length_mean"}
    )

    return data_avg_per_user


def merge_matrix_avgs_friend_list(
    matrix: pd.DataFrame, averages: pd.DataFrame, friend_lists: pd.DataFrame
):
    # Copy useful columns from Dataframe and fix time format
    merged = matrix[["user_id", "video_id", "watch_ratio", "video_length", "time"]]
    merged["time"] = pd.to_datetime(merged["time"])

    # Merge previously calculated averages
    merged = merged.merge(averages, on="user_id", how="left")

    # Merge friend lists
    merged = merged.merge(friend_lists, on="user_id", how="left")

    # Fix friend lists missing values
    merged["friend_list"] = merged["friend_list"].apply(
        lambda x: [] if isinstance(x, float) and np.isnan(x) else x
    )

    return merged


def get_friends_data(df: pd.DataFrame):
    # Copy Dataframe
    df_with_friends_data = df.copy()

    # Sort Dataframe by time
    df_with_friends_data = df_with_friends_data.sort_values("time").reset_index(
        drop=True
    )

    # Compute prior video mean for each video (exclude current row using shift)
    df_with_friends_data["video_watch_ratio_prior_mean"] = df_with_friends_data.groupby(
        "video_id"
    )["watch_ratio"].transform(  # For each video's watch_ratios
        lambda x: x.shift().expanding().mean()
    )  # Shift rows to exclude current one, and then perform the mean of every watch_ratio before the current one

    return df_with_friends_data


def get_friends_watch_ratio(df: pd.DataFrame):
    df_with_friends_data = get_friends_data(df)

    # Explode friends to work with numbers instead of lists and rename column
    df_exploded = df_with_friends_data.explode("friend_list").rename(
        columns={"friend_list": "friend_id"}
    )

    # Copy main previously calculated features and rename columns
    friend_history = df_with_friends_data[
        ["user_id", "video_id", "time", "watch_ratio"]
    ].copy()
    friend_history.columns = [
        "friend_id",
        "video_id",
        "friend_time",
        "friend_watch_ratio",
    ]

    # Merge friends data
    merged = df_exploded.merge(friend_history, on=["friend_id", "video_id"], how="left")
    # Filter only prior data
    merged = merged[merged["friend_time"] < merged["time"]]

    # Compute friend's watch ratio mean for each interaction
    friend_means = (
        merged.groupby(["user_id", "video_id", "time"])["friend_watch_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"friend_watch_ratio": "friend_watch_ratio_prior_mean"})
    )

    # Merge back friend means and fill missing values with video means if possible
    final_data = df_with_friends_data.merge(
        friend_means, on=["user_id", "video_id", "time"], how="left"
    )
    final_data["watch_ratio_prior_mean"] = (
        final_data["friend_watch_ratio_prior_mean"]
        .fillna(final_data["video_watch_ratio_prior_mean"])
        .fillna(0)
    )

    # Drop temporary columns
    final_data = final_data.drop(
        columns=[
            "friend_watch_ratio_prior_mean",
            "video_watch_ratio_prior_mean",
            "friend_list",
        ]
    )

    return final_data
