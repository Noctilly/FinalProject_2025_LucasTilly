import pandas as pd
import numpy as np


def compute_averages(df: pd.DataFrame):
    # Copy Dataframe and fix time format
    data_avg_per_user = df.copy()

    # Get hour from time column
    # data_avg_per_user["hour"] = data_avg_per_user["time"].dt.hour

    # Compute averages of all numeric values for each user
    data_avg_per_user = (
        data_avg_per_user.groupby("user_id").mean(numeric_only=True).reset_index()
    )

    # Drop columns which don't make sense as averages
    data_avg_per_user = data_avg_per_user.drop(columns=["video_id", "like"])

    # Rename columns for clarity
    data_avg_per_user = data_avg_per_user.rename(
        columns={
            "watch_ratio": "user_watch_ratio_mean",
            "video_length": "user_video_length_mean",
        }
    )

    return data_avg_per_user


def merge_matrix_avgs_friend_list(
    matrix: pd.DataFrame, averages: pd.DataFrame, friend_lists: pd.DataFrame
):
    # Copy useful columns from Dataframe and fix time format
    merged = matrix[
        ["user_id", "video_id", "watch_ratio", "like", "video_length", "time"]
    ].copy()

    # Merge previously calculated averages
    merged = merged.merge(averages, on="user_id", how="left")

    # Merge friend lists
    merged = merged.merge(friend_lists, on="user_id", how="left")

    # Fix friend lists missing values
    merged["friend_list"] = merged["friend_list"].apply(
        lambda x: [] if isinstance(x, float) and np.isnan(x) else x
    )

    return merged


def get_friends_watch_ratio(df: pd.DataFrame, video_features: pd.DataFrame):
    # df_with_friends_data = get_friends_data(df)

    # Explode friends to work with numbers instead of lists and rename column
    df_exploded = df.explode("friend_list").rename(columns={"friend_list": "friend_id"})

    # Copy main previously calculated features and rename columns
    friend_history = df[["user_id", "video_id", "watch_ratio"]].copy()
    friend_history.columns = [
        "friend_id",
        "video_id",
        "friend_watch_ratio",
    ]

    # Merge friends data
    merged = df_exploded.merge(friend_history, on=["friend_id", "video_id"], how="left")
    # Filter only prior data
    # merged = merged[merged["friend_time"] < merged["time"]]

    # Compute friend's watch ratio mean for each interaction
    friend_means = (
        merged.groupby(["user_id", "video_id"])["friend_watch_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"friend_watch_ratio": "friend_watch_ratio_mean"})
    )

    # Merge back friend means and fill missing values with video means if possible
    final_data = df.merge(friend_means, on=["user_id", "video_id"], how="left")
    final_data["friend_watch_ratio_mean"] = final_data[
        "friend_watch_ratio_mean"
    ]  # .fillna(video_features["average_watch_ratio"].mean())

    # Drop temporary columns
    final_data = final_data.drop(
        columns=[
            "friend_list",
        ]
    )

    return final_data


def get_video_feat_category_avg_per_user(
    matrix: pd.DataFrame,
    avg_user_feat: pd.DataFrame,
    avg_user_category: pd.DataFrame,
    video_feat: pd.DataFrame,
    video_category: pd.DataFrame,
):
    user_video_info = matrix[["user_id", "video_id"]].copy()

    # Handle feat
    user_video_info = user_video_info.merge(video_feat, on="video_id", how="left")
    user_video_info = user_video_info.explode("feat")
    user_video_info = user_video_info.merge(
        avg_user_feat, on=["user_id", "feat"], how="left"
    )
    user_video_info = user_video_info.drop(columns="feat")

    # For each (user, video) couple, compute the average of the user's watch ratios on this video's tags
    user_video_info["user_feat_watch_ratio_mean"] = user_video_info.groupby(
        ["user_id", "video_id"]
    )["user_feat_watch_ratio_mean"].transform("mean")

    user_video_info = user_video_info.drop_duplicates()

    # Handle category
    user_video_info = user_video_info.merge(video_category, on="video_id", how="left")
    user_video_info = user_video_info.merge(
        avg_user_category, on=["user_id", "first_level_category_id"], how="left"
    )
    user_video_info = user_video_info.drop(columns="first_level_category_id")

    # user_id, video_id, user_feat_watch_ratio_mean, user_category_watch_ratio_mean
    return user_video_info
