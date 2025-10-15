import logging
from pathlib import Path

import numpy as np
import polars as pl  # noqa: PLC0415
from hydra.utils import instantiate
from omegaconf import DictConfig
from rbyte.io import (  # noqa: PLC0415
    DataFrameConcater,
    DataFrameDuckDbQuery,
    DataFrameGroupByDynamic,
)

import rbyte
from rbyte.config import BaseModel


def reye_monkey_patcher(
    reye_pred_path: Path, gby_config: DataFrameGroupByDynamic, query: str | None = None
) -> pl.DataFrame:
    """Generate rbyte dataset from reye format predictions.
    TODO: Should be in yaml as a target "pipeline" with
    iteration over drives as th only code component.
    Maybe this can also be done via yaml.
    """  # noqa: DOC201

    parquet_files = Path(reye_pred_path).glob("*.parquet")
    df = pl.read_parquet(list(parquet_files))
    columns = (
        ReyeColumns() if ReyeColumns().col_brake_gt in df.columns else ReyeColumnRmind()
    )
    df = (
        df.select(
            pl.col(columns.col_frame_idx).arr.last(),
            pl.col(columns.col_drive_id),
            pl.col(columns.col_gas_pred).arr.last(),
            pl.col(columns.col_brake_pred).arr.last(),
            pl.col(columns.col_steering_pred).arr.last(),
            pl.col(columns.col_gas_gt).arr.last().arr.last(),
            pl.col(columns.col_brake_gt).arr.last().arr.last(),
            pl.col(columns.col_steering_gt).arr.last().arr.last(),
            pl.col(columns.col_timestamp).arr.last(),
        )
        if df[columns.col_frame_idx].dtype == pl.Array
        else df
    )

    gby = gby_config if isinstance(gby_config, DataFrameGroupByDynamic) else None
    dbquery = DataFrameDuckDbQuery()
    input_col = "batch/meta/input_id"
    drives = df[input_col].unique().sort().to_list()

    samples = []
    valid_drives = []
    for drive in drives:
        logging.info(f"Processing drive: {drive}")  # noqa: G004, LOG015

        selected = dbquery(
            query=query,
            samples=(
                df.filter(pl.col(input_col) == drive)
                if gby is None
                else gby(df.filter(pl.col(input_col) == drive))
            ),
        )
        if len(selected) == 0:
            continue
        samples.append(selected)
        valid_drives.append(drive)

    cctr = DataFrameConcater(
        key_column="input_id",
    )
    samples = cctr(
        keys=valid_drives,
        values=samples,
    )
    logging.info(  # noqa: LOG015
        f"built samples height={samples.height}, size={samples.estimated_size(unit := 'gb'):.3f}",  # noqa: G004
    )
    return samples


class ReyeColumns(BaseModel):
    col_brake_gt: str = "predictions/policy/ground_truth/continuous/brake_pedal"
    col_gas_gt: str = "predictions/policy/ground_truth/continuous/gas_pedal"
    col_steering_gt: str = "predictions/policy/ground_truth/continuous/steering_angle"
    col_brake_pred: str = "predictions/policy/prediction_value/continuous/brake_pedal"
    col_gas_pred: str = "predictions/policy/prediction_value/continuous/gas_pedal"
    col_steering_pred: str = (
        "predictions/policy/prediction_value/continuous/steering_angle"
    )
    col_frame_idx: str = "batch/data/meta/ImageMetadata.cam_front_left/frame_idx"
    col_drive_id: str = "batch/meta/input_id"
    col_timestamp: str = "batch/data/meta/ImageMetadata.cam_front_left/time_stamp"


class ReyeColumnRmind(BaseModel):
    col_brake_gt: str = "policy/ground_truth/continuous/brake_pedal"
    col_gas_gt: str = "policy/ground_truth/continuous/gas_pedal"
    col_steering_gt: str = "policy/ground_truth/continuous/steering_angle"
    col_brake_pred: str = "policy/prediction_value/continuous/brake_pedal"
    col_gas_pred: str = "policy/prediction_value/continuous/gas_pedal"
    col_steering_pred: str = "policy/prediction_value/continuous/steering_angle"
    col_frame_idx: str = "batch/data/meta/ImageMetadata.cam_front_left/frame_idx"
    col_drive_id: str = "batch/meta/input_id"
    col_timestamp: str = "batch/data/meta/ImageMetadata.cam_front_left/time_stamp"


class RbyteColumns(BaseModel):
    col_frame_idx: str = "meta/ImageMetadata.cam_front_left/frame_idx"
    col_drive_id: str = "__input_id"
    col_gas: str = "meta/VehicleMotion/gas_pedal_normalized"
    col_brake: str = "meta/VehicleMotion/brake_pedal_normalized"
    col_steering: str = "meta/VehicleMotion/steering_angle_normalized"
    col_timestamp: str = "meta/ImageMetadata.cam_front_left/time_stamp"


def reye_eval_dataset(
    reye_pred_path: Path, change_sampled: pl.DataFrame
) -> pl.DataFrame:
    """Generate rbyte dataset from reye format predictions.
    TODO: Should be in yaml as a target "pipeline" with
    iteration over drives as th only code component.
    Maybe this can also be done via yaml.
    """  # noqa: DOC201
    import polars as pl  # noqa: PLC0415
    from rbyte.io import (  # noqa: PLC0415
        DataFrameConcater,
    )

    reye_columns = ReyeColumns()
    rbyte_columns = RbyteColumns()
    parquet_files = Path(reye_pred_path).glob("*.parquet")
    df = pl.read_parquet(list(parquet_files))
    drives = df[reye_columns.col_drive_id].unique().sort().to_list()
    drives_change = change_sampled[rbyte_columns.col_drive_id].unique().sort().to_list()
    drives = [drive for drive in drives if drive in drives_change]
    if reye_columns.col_brake_gt not in df.columns:
        reye_columns = ReyeColumnRmind()
    samples = []
    valid_drives = []
    for drive in drives:
        logging.info(f"Processing drive: {drive}")  # noqa: G004, LOG015
        df_change = change_sampled.filter(pl.col(rbyte_columns.col_drive_id) == drive)
        if len(df_change) == 0:
            logging.warning(f"No changes for drive {drive}, skipping")  # noqa: G004, LOG015
            continue
        frame_ids = np.stack(df_change[rbyte_columns.col_frame_idx])[:, -1]
        df_pred = df.filter(pl.col(reye_columns.col_drive_id) == drive)
        frames_pred = np.stack(df_pred[reye_columns.col_frame_idx])
        inds = np.where(np.isin(frames_pred, frame_ids, assume_unique=True))[0]  # noqa: PLC3001
        samples.append(
            df_pred.filter(
                pl.col(reye_columns.col_frame_idx).arr.last().is_in(frame_ids)
            )
            if frames_pred.ndim > 1
            else df_pred.filter(pl.col(reye_columns.col_frame_idx).is_in(frame_ids))
        )
        valid_drives.append(drive)

    cctr = DataFrameConcater(
        key_column="input_id",
    )
    samples = cctr(
        keys=valid_drives,
        values=samples,
    )
    logging.info(  # noqa: LOG015
        f"built samples height={samples.height}, size={samples.estimated_size(unit := 'gb'):.3f}",  # noqa: G004
    )
    return samples


def filter_rmind_by_copycat(
    rmind_samples: pl.DataFrame, copycat_samples: pl.DataFrame
) -> pl.DataFrame:
    reye_columns = ReyeColumns()
    drives_copycat = (
        copycat_samples[reye_columns.col_drive_id].unique().sort().to_list()
    )
    rmind_columns = ReyeColumnRmind()
    drives_rmind = rmind_samples[rmind_columns.col_drive_id].unique().sort().to_list()
    drives_rmind = [drive for drive in drives_rmind if drive in drives_copycat]
    filtered_samples = []
    for drive in drives_rmind:
        df_eval = copycat_samples.filter(pl.col(reye_columns.col_drive_id) == drive)
        df_rmind = rmind_samples.filter(pl.col(rmind_columns.col_drive_id) == drive)
        copycat_times = df_eval[reye_columns.col_timestamp]
        rmind_times = df_rmind[rmind_columns.col_timestamp].arr.last().to_physical()
        closest_indices = []
        for copy_times in copycat_times:
            d = (copy_times - rmind_times).abs()
            closest = d.arg_min()
            closest_indices.append(closest)
        filtered_samples.append(df_rmind[closest_indices])
    from rbyte.io import (  # noqa: PLC0415
        DataFrameConcater,
    )

    cctr = DataFrameConcater(
        key_column="drive_id",
    )
    return cctr(
        keys=drives_rmind,
        values=filtered_samples,
    )
