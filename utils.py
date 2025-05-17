import pandas as pd


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
    data_avg_per_user = data_avg_per_user.drop(columns=["video_id"])

    # Rename columns for clarity
    data_avg_per_user = data_avg_per_user.rename(
        columns={
            "watch_ratio": "user_watch_ratio_mean",
            "video_duration": "user_video_duration_mean",
        }
    )

    return data_avg_per_user


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
