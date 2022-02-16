import argparse
import logging
from bnnrul.cmapss.preprocessing import generate_parquet, generate_lmdb

if __name__ == "__main__":
    logFormatter = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=logFormatter, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Prepare lmdb")
    args = parser.parse_args()
    args.out_path = "data/cmapss/"
    args.normalization = "min-max"
    args.validation = 0.2
    args.subsets = ["FD001"]  # "FD002", "FD003", "FD004"],
    # win_length={"FD001": 30, "FD002": 20, "FD003": 30, "FD004": 15}, #variable length per subset only ok for LSTM models
    args.win_length = dict(
        {
            1: 18,
            2: 18,
            3: 18,
            4: 18,
        }
    )  # variable length per subset fixed for Linear/Conv models
    args.win_step = 1
    args.settings = list(["setting_1", "setting_2", "setting_3"])
    args.features = list(
        [
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_7",
            "sensor_8",
            "sensor_9",
            "sensor_11",
            "sensor_12",
            "sensor_13",
            "sensor_14",
            "sensor_15",
            "sensor_17",
            "sensor_20",
            "sensor_21",
        ]
    )
    args.bits = 32

    generate_parquet(args)
    generate_lmdb(args, datasets=["train"])
