import logging
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Union

import numpy as np
import pandas as pd
from ..utils.lmdb_utils import create_lmdb
from tqdm.autonotebook import tqdm


def generate_parquet(args):
    for subset in args.subsets:
        logging.info("**** %s ****" % subset)
        logging.info("normalization = " + args.normalization)
        logging.info("validation = " + str(args.validation))

        # read .zip file into memory
        with zipfile.ZipFile(f"{args.out_path}/CMAPSSData.zip") as zip_file:
            file_train = zip_file.open("train_" + subset + ".txt")
            file_test = zip_file.open("test_" + subset + ".txt")
            file_rul = zip_file.open("RUL_" + subset + ".txt")

        print("Extracting dataframes...")
        df_train, df_val, df_test = extract_dataframes(
            file_train=file_train,
            file_test=file_test,
            file_rul=file_rul,
            subset=subset,
            validation=args.validation,
        )

        print("Generating parquet files...")
        path = Path(args.out_path, "parquet")
        path.mkdir(exist_ok=True)
        for df, prefix in zip([df_train, df_val, df_test], ["train", "val", "test"]):
            if isinstance(df, pd.DataFrame):
                df.to_parquet(f"{path}/{prefix}_{subset}.parquet")


def extract_dataframes(
    file_train, file_test, file_rul, subset="FD001", validation=0.00
):
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
    rul = np.asarray(file_rul.readlines(), dtype=np.int32)
    cumul = []
    for traj_id, traj in df_test.groupby("trajectory_id"):
        cumul.append(traj.assign(rul=rul[traj_id - 1] - (traj.index - traj.index[0])))
    df_test = pd.concat(cumul)

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
    # drop operating_modes
    # df = df.drop(
    #     ["setting_" + str(i + 1) for i in range(n_operational_settings)], axis=1
    # )
    # drop sensors which are useless according to the literature
    to_drop = [1, 5, 6, 10, 16, 18, 19]
    df = df.drop(["sensor_" + str(d) for d in to_drop], axis=1)

    return df


def generate_lmdb(args, datasets=["train", "val", "test"]):
    lmdb_dir = Path(f"{args.out_path}/lmdb")
    lmdb_dir.mkdir(exist_ok=True)
    for ds in datasets:
        print(f"Generating {ds} lmdb ...")
        filelist = list(Path(f"{args.out_path}/parquet").glob(f"{ds}*.parquet"))
        if filelist is not None:
            feed_lmdb(Path(f"{lmdb_dir}/{ds}.lmdb"), filelist, args)


def feed_lmdb(output_lmdb: Path, filelist: List[Path], args) -> None:
    patterns: Dict[str, Callable[[Line], Union[bytes, np.ndarray]]] = {
        "{}": (
            lambda line: line.data.astype(np.float32) if args.bits == 32 else line.data
        ),
        "ds_id_{}": (lambda line: "{}".format(line.ds_id).encode()),
        "traj_id_{}": (lambda line: "{}".format(line.traj_id).encode()),
        "win_id_{}": (lambda line: "{}".format(line.win_id).encode()),
        "settings_{}": (
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
        n_features=len(args.features),
        bits=args.bits,
    )


class Line(NamedTuple):
    ds_id: int
    traj_id: int
    win_id: int
    settings: np.ndarray
    data: np.ndarray
    rul: int


def process_files(filelist: List[Path], args) -> Iterator[Line]:
    for filename in tqdm(filelist):
        df = pd.read_parquet(filename)
        args.subset = int(str(filename.stem).split("_")[-1][-1])
        yield from process_dataframe(df, args)
        del df


def process_dataframe(df: pd.DataFrame, args) -> Iterator[Line]:
    win_length = (
        args.win_length[args.subset]
        if isinstance(args.win_length, dict)
        else args.win_length
    )
    for traj_id, traj in df.groupby("trajectory_id"):
        for i, sl in enumerate(make_slice(traj.shape[0], win_length, args.win_step)):
            yield Line(
                ds_id=args.subset,
                traj_id=traj_id,
                win_id=i,
                settings=traj[args.settings].iloc[sl].unstack().values,
                data=traj[args.features].iloc[sl].unstack().values,
                rul=traj["rul"].iloc[sl].iloc[-1],
            )


def make_slice(total: int, size: int, step: int) -> Iterator[slice]:
    for i in range(total // step):
        if i * step + size < total:
            yield slice(i * step, i * step + size)
        if i * step + size >= total:
            yield slice(total - size, total)
            return


class MinMaxAggregate:
    def __init__(self, args):
        self.args = args

    def feed(self, line: Line, i: int) -> None:
        n_features = len(self.args.features)
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
