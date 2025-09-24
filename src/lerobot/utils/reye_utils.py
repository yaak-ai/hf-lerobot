import polars as pl
from torch.utils.data import DataLoader


def _create_reye_columns(actions, is_without_clip: bool) -> list[pl.Expr]:
    cols = [
        pl.col("__input_id").alias("batch/meta/input_id").cast(pl.String),
    ]

    if is_without_clip:
        # Single timestamp model (without clip)
        cols.extend([
            pl.col("meta/ImageMetadata.cam_front_left/time_stamp")
            .to_physical()
            .alias("batch/data/meta/ImageMetadata.cam_front_left/time_stamp"),
            pl.col("meta/ImageMetadata.cam_front_left/frame_idx").alias(
                "batch/data/meta/ImageMetadata.cam_front_left/frame_idx"
            ),
        ])
    else:
        cols.extend([
            pl.col("meta/ImageMetadata.cam_front_left/time_stamp")
            .list.last()
            .to_physical()
            .alias("batch/data/meta/ImageMetadata.cam_front_left/time_stamp"),
            pl.col("meta/ImageMetadata.cam_front_left/frame_idx")
            .list.last()
            .alias("batch/data/meta/ImageMetadata.cam_front_left/frame_idx"),
        ])

    cols.extend([
        pl.col("meta/VehicleMotion/brake_pedal_normalized")
        .list.last()
        .alias("predictions/policy/ground_truth/continuous/brake_pedal"),
        pl.col("meta/VehicleMotion/gas_pedal_normalized")
        .list.last()
        .alias("predictions/policy/ground_truth/continuous/gas_pedal"),
        pl.col("meta/VehicleMotion/steering_angle_normalized")
        .list.last()
        .alias("predictions/policy/ground_truth/continuous/steering_angle"),
        pl.lit(0).alias("predictions/policy/ground_truth/discrete/turn_signal"),
        pl.Series(
            values=actions[:, 0],
            name="predictions/policy/prediction_value/continuous/gas_pedal",
        ),
        pl.Series(
            values=actions[:, 1],
            name="predictions/policy/prediction_value/continuous/brake_pedal",
        ),
        pl.Series(
            values=actions[:, -1],
            name="predictions/policy/prediction_value/continuous/steering_angle",
        ),
        pl.lit(1)
        .alias("predictions/policy/prediction_value/discrete/turn_signal")
        .cast(pl.Float32),
    ])

    for action in ["gas_pedal", "brake_pedal", "steering_angle"]:
        cols.extend([
            pl.lit(0)
            .alias(f"predictions/policy/prediction_std/continuous/{action}")
            .cast(pl.Float32),
            pl.lit(1)
            .alias(f"predictions/policy/prediction_probs/continuous/{action}")
            .cast(pl.Float32),
            pl.lit(1)
            .alias(f"predictions/policy/score_logprob/continuous/{action}")
            .cast(pl.Float32),
        ])

    cols.extend([
        pl.lit(0)
        .alias("predictions/policy/prediction_std/discrete/turn_signal")
        .cast(pl.Float32),
        pl.lit(1)
        .alias("predictions/policy/prediction_probs/discrete/turn_signal")
        .cast(pl.Float32),
        pl.lit(1)
        .alias("predictions/policy/score_logprob/discrete/turn_signal")
        .cast(pl.Float32),
    ])

    return cols


def create_reye_df(eval_dataloader: DataLoader, actions, is_without_clip: bool) -> pl.DataFrame:
    cols = _create_reye_columns(actions, is_without_clip)
    return eval_dataloader.dataset.samples.select(cols).with_columns([
        (
            (
                pl.col("predictions/policy/prediction_value/continuous/gas_pedal")
                - pl.col("predictions/policy/ground_truth/continuous/gas_pedal")
            ).abs()
        ).alias("predictions/policy/score_l1/continuous/gas_pedal"),
        (
            (
                pl.col("predictions/policy/prediction_value/continuous/brake_pedal")
                - pl.col("predictions/policy/ground_truth/continuous/brake_pedal")
            ).abs()
        ).alias("predictions/policy/score_l1/continuous/brake_pedal"),
        (
            (
                pl.col("predictions/policy/prediction_value/continuous/steering_angle")
                - pl.col("predictions/policy/ground_truth/continuous/steering_angle")
            ).abs()
        ).alias("predictions/policy/score_l1/continuous/steering_angle"),
        pl.lit(0)
        .alias("predictions/policy/score_l1/discrete/turn_signal")
        .cast(pl.Float32),
    ])
