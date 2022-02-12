import zipfile
import argparse
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Union

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from mltbox.utils.lmdb_utils import create_lmdb


class Line(NamedTuple):
    id: int
    subset: int
    settings: np.ndarray
    data: np.ndarray
    rul: int


class MinMaxAggregate:
    def __init__(self, args):
        self.args = args

    def feed(self, line: Line, i: int) -> None:
        n_features = len(args.features) - 1
        min_, max_ = (
            line.data.reshape(n_features, -1).T,
            line.data.reshape(n_features, -1).T,
        )

        if i == 0:
            self.min_, self.max_ = min_.min(0), max_.max(0)
        else:
            self.min_ = np.min([self.min_, min_.min(0)], axis=0)
            self.max_ = np.max([self.max_, max_.max(0)], axis=0)

    def get(self) -> Dict[str, Union[np.ndarray, bytes]]:
        return {
            "min_sample": self.min_.astype(
                np.float32 if self.args.bits == 32 else np.float64
            ),
            "max_sample": self.max_.astype(
                np.float32 if self.args.bits == 32 else np.float64
            ),
        }


def make_slice(total: int, size: int, step: int) -> Iterator[slice]:
    for i in range(total // step):
        if i * step + size < total:
            yield slice(i * step, i * step + size)
        if i * step + size >= total:
            yield slice(total - size, total)
            return


def extract_dataframes(file_train, file_test, subset="FD001", validation=0.00):
    """Extract train, validation and test dataframe from source file.

    Parameters
    ----------
    file_train : str
        Training samples file.
    file_test : str
        Test samples file.
    subset: str, optional
        Subset. Either 'FD001' or 'FD002' or 'FD003' or 'FD004'.
    validation : float, optional
        Ratio of training samples to hold out for validation.

    Returns
    -------
    (DataFrame, DataFrame, DataFrame)
        Train dataframe, validation dataframe, test dataframe.
    """
    assert subset in ["FD001", "FD002", "FD003", "FD004"], (
        "'subset' must be either 'FD001' or 'FD002' or 'FD003' or 'FD004', got '"
        + subset
        + "'."
    )

    assert 0 <= validation <= 1, (
        "'validation' must be a value within [0, 1], got %.2f" % validation + "."
    )

    df = _load_data_from_file(file_train, subset=subset)

    # group by trajectory
    grouped = df.groupby("trajectory_id")

    df_train = []
    df_validation = []
    for _, traj in grouped:
        traj = traj.assign(rul=traj.index[-1] - traj.index)
        # randomize train/validation splitting
        if np.random.rand() <= (validation + 0.1) and len(df_validation) < round(
            len(grouped) * validation
        ):
            df_validation.append(traj)
        else:
            df_train.append(traj)

    # print info
    print("Number of training trajectories = " + str(len(df_train)))
    print("Number of validation trajectories = " + str(len(df_validation)))

    df_train = pd.concat(df_train)

    if len(df_validation) > 0:
        df_validation = pd.concat(df_validation)

    df_test = _load_data_from_file(file_test, subset=subset)

    print("Done.")
    return df_train, df_validation, df_test


def _load_data_from_file(file, subset="FD001"):
    """Load data from source file into a dataframe.

    Parameters
    ----------
    file : str
        Source file.
    subset: str, optional
        Subset. Either 'FD001' or 'FD002' or 'FD003' or 'FD004'.

    Returns
    -------
    DataFrame
        Data organized into a dataframe.
    """
    assert subset in ["FD001", "FD002", "FD003", "FD004"], (
        "'subset' must be either 'FD001' or 'FD002' or 'FD003' or 'FD004', got '"
        + subset
        + "'."
    )

    n_operational_settings = 3
    n_sensors = 21

    # read csv
    df = pd.read_csv(file, sep=" ", header=None, index_col=False).fillna(method="bfill")
    df = df.dropna(axis="columns", how="all")

    assert (
        df.shape[1] == n_operational_settings + n_sensors + 2
    ), "Expected %d columns, got %d." % (
        n_operational_settings + n_sensors + 2,
        df.shape[1],
    )

    df.columns = (
        ["trajectory_id", "t"]
        + ["setting_" + str(i + 1) for i in range(n_operational_settings)]
        + ["sensor_" + str(i + 1) for i in range(n_sensors)]
    )

    # drop t
    df = df.drop(["t"], axis=1)

    # if subset in ["FD001", "FD003"]:
    #     # drop operating_modes
    #     df = df.drop(
    #         ["setting_" + str(i + 1) for i in range(n_operational_settings)], axis=1
    #     )

    # drop sensors which are useless according to the literature
    to_drop = [1, 5, 6, 10, 16, 18, 19]
    df = df.drop(["sensor_" + str(d) for d in to_drop], axis=1)

    df = df.assign(subset=int(subset[-1]))

    return df


def process_dataframe(df: pd.DataFrame, args) -> Iterator[Line]:
    for system in range(1, len(args.features.columns) + 1):
        if hasattr(args, "system") and args.system != system:
            continue
        sys_features = args.features[str(system)]
        chunk = (
            df.query(f"{sys_features[1]} == {sys_features[1]}")[sys_features]
            .reset_index()
            .drop(columns=["index"])
        )
        yield from process_chunk(chunk, tail, flight, system, args)


def process_chunk(chunk, tail, flight, system, args):
    if chunk.shape[0] >= args.win_length:
        for sl in make_slice(chunk.shape[0], args.win_length, args.win_step):
            data = chunk.iloc[sl, 1:].unstack().values
            yield Line(
                timestamp=chunk.timestamp.iloc[sl.start],
                data=data,
                tail=tail,
                system=system,
                flight=flight,
            )
    else:
        args.excluded.append([tail, system, flight, "win_len"])


def process_files(file_list: List[Path], args) -> Iterator[Line]:
    for file in tqdm(file_list):
        yield from process_dataframe(file, args)


def feed_lmdb(output_lmdb: Path, filelist: List[Path], args) -> None:
    patterns: Dict[str, Callable[[Line], Union[bytes, np.ndarray]]] = {
        "{}": (
            lambda line: line.data.astype(np.float32) if args.bits == 32 else line.data
        ),
        "id_{}": (lambda line: "{}".format(line.id).encode()),
        "subset_{}": (lambda line: "{}".format(line.subset).encode()),
        "settings{}": (
            lambda line: line.settings.astype(np.float32)
            if args.bits == 32
            else line.settings
        ),
        "rul_{}": (lambda line: "{}".format(line.rul).encode()),
    }

    return create_lmdb(
        filename=output_lmdb,
        iterator=process_files(filelist, args),
        patterns=patterns,
        aggs=[MinMaxAggregate(args)],
        win_length=args.win_length,
        n_features=len(args.features) - 1,
        bits=args.bits,
    )


def generate_parquet(args):
    for subset, window in [("FD001", 30), ("FD002", 20), ("FD003", 30), ("FD004", 15)]:
        print("**** %s ****" % subset)
        print("normalization = " + args.normalization)
        print("window = " + str(window))
        print("validation = " + str(args.validation))

        # read .zip file into memory
        with zipfile.ZipFile("data/cmapss/CMAPSSData.zip") as zip_file:
            file_train = zip_file.open("train_" + subset + ".txt")
            file_test = zip_file.open("test_" + subset + ".txt")
            file_rul = zip_file.open("RUL_" + subset + ".txt")

        print("Extracting dataframes...")
        dataframes = extract_dataframes(
            file_train=file_train,
            file_test=file_test,
            file_rul=file_rul,
            subset=subset,
            validation=args.validation,
        )

        print("Generating parquet files...")
        path = os.path.join(args.out_path, "parquet")
        if not os.path.exists(path):
            os.makedirs(path)
        for df, prefix in zip(dataframes, ["train", "val", "test"]):
            df.to_parquet(f"{path}/{prefix}_{subset}.parquet")


if __name__ == "__main__":
    logFormatter = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=logFormatter, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Prepare lmdb")
    args = parser.parse_args()
    args.out_path = "data/cmapss/"
    args.normalization = "min-max"
    args.validation = 0.00

    generate_parquet(args)
    # feed_lmdb(output_lmdb, args)
