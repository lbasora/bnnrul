import argparse
from pathlib import Path

import pytorch_lightning as pl
from bnnrul.cmapss.dataset import CMAPSSDataModule
from bnnrul.cmapss.models import CMAPSSModel, get_checkpoint
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out_path", type=str, default="./results/cmapss")
    parser.add_argument("--data_path", type=str, default="./data/cmapss")
    parser.add_argument("--scn", type=str, default="deterministic")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser = CMAPSSModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.gpus = [args.gpu]

    data = CMAPSSDataModule(args.data_path, args.batch_size)
    model = CMAPSSModel(data.win_length, data.n_features, args.net)

    checkpoint_dir = Path(f"{args.out_path}/{args.scn}/checkpoints/{args.net}/")
    checkpoint_file = get_checkpoint(checkpoint_dir)
    monitor = "loss/val"
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, monitor=monitor)
    earlystopping_callback = EarlyStopping(monitor=monitor, patience=20)

    trainer = pl.Trainer.from_argparse_args(
        args,
        # ckpt_path=checkpoint,
        resume_from_checkpoint=checkpoint_file,
        logger=pl.loggers.TensorBoardLogger(
            f"{args.out_path}/{args.scn}/lightning_logs/{args.net}",
            name=f"{args.net}",
            default_hp_metric=False,
        ),
        callbacks=[
            checkpoint_callback,
            earlystopping_callback,
        ],
    )
    trainer.fit(model, data)
