import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Tuple

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import optuna
from optuna.pruners import MedianPruner
from data.ordinances import BatchDataModule, NERDataModule

from log import logger
from model.ner_model import NERBaseAnnotator


def get_out_filename(out_dir, model, prefix):
    model_name = os.path.basename(model)
    model_name = model_name[: model_name.rfind(".")]
    return "{}/{}_base_{}.tsv".format(out_dir, prefix, model_name)


def write_eval_performance(eval_performance, out_file):
    outstr = ""
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ["results", "predictions"]:
                continue
            outstr = outstr + "{}\t{}\n".format(k, out_[k])
            added_keys.add(k)

    open(out_file, "wt").write(outstr)
    logger.info("Finished writing evaluation performance for {}".format(out_file))


def load_model(model_file: str, label2id: Mapping[str, int], stage="prediction"):
    if ~os.path.isfile(model_file):
        model_file = get_models_for_evaluation(model_file)

    hparams_file = model_file[: model_file.rindex("checkpoints/")] + "/hparams.yaml"
    model = NERBaseAnnotator.load_from_checkpoint(
        model_file, hparams_file=hparams_file, label2id=label2id, stage=stage
    )
    model.stage = stage
    return model, model_file


def save_model(trainer, out_dir, model_name="", timestamp=None):
    version = trainer.logger.version if trainer.logger else 0
    out_dir = out_dir / ("lightning_logs/version_" + str(version) + "/checkpoints/")
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir / (model_name + "_timestamp_" + str(timestamp) + "_final.ckpt")
    trainer.save_checkpoint(outfile, weights_only=True)

    logger.info("Stored model {}.".format(outfile))
    return outfile


def train_model(
    model: NERBaseAnnotator,
    datamodule: NERDataModule,
    epochs: int,
    out_dir: Path,
    grad_norm: float,
    reload_every_epoch: bool,
    es_callback: Optional[EarlyStopping] = None,
):
    seed_everything(42)
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if es_callback is not None:
        callbacks.append(es_callback)
    trainer = Trainer(
        max_epochs=epochs,
        default_root_dir=out_dir,
        callbacks=callbacks,
        gradient_clip_algorithm="norm",
        gradient_clip_val=grad_norm,
        reload_dataloaders_every_n_epochs=1 if reload_every_epoch else 0,
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer


def train_batch(
    model: NERBaseAnnotator,
    datamodule: BatchDataModule,
    epochs: int,
    out_dir: Path,
    grad_norm: float,
    metric: str = "val_MD@F1",
) -> Trainer:
    es_callback = EarlyStopping(
        monitor=metric, min_delta=0.001, patience=3, verbose=True, mode="max"
    )
    return train_model(
        model,
        datamodule,
        epochs,
        out_dir,
        grad_norm,
        reload_every_epoch=False,
        es_callback=es_callback,
    )


def get_models_for_evaluation(path):
    if "checkpoints" not in path:
        path = path + "/checkpoints/"
    model_files = list_files(path)
    models = [f for f in model_files if f.endswith("final.ckpt")]

    return models[0] if len(models) != 0 else None


def list_files(in_dir):
    files = []
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files
