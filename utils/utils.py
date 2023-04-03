import os
from pathlib import Path
import time
from typing import Any, Mapping, Tuple

import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import optuna
from optuna.pruners import MedianPruner
from data.ordinances import OrdinancesDataModule

from log import logger
from model.callbacks import PyTorchLightningPruningCallback
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


def load_model(model_file, tag_to_id=None, stage="test"):
    if ~os.path.isfile(model_file):
        model_file = get_models_for_evaluation(model_file)

    hparams_file = model_file[: model_file.rindex("checkpoints/")] + "/hparams.yaml"
    model = NERBaseAnnotator.load_from_checkpoint(
        model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id
    )
    model.stage = stage
    return model, model_file


def save_model(trainer, out_dir, model_name="", timestamp=None):
    out_dir = out_dir / (
        "lightning_logs/version_" + str(trainer.logger.version) + "/checkpoints/"
    )
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir / (model_name + "_timestamp_" + str(timestamp) + "_final.ckpt")
    trainer.save_checkpoint(outfile, weights_only=True)

    logger.info("Stored model {}.".format(outfile))
    return outfile


def train_model(
    model: NERBaseAnnotator,
    datamodule: OrdinancesDataModule,
    epochs: int,
    out_dir: Path,
    metric: str = "val_MD@F1",
) -> Trainer:
    seed_everything(42)
    es_callback = EarlyStopping(
        monitor=metric, min_delta=0.001, patience=3, verbose=True, mode="max"
    )
    lr_logger = LearningRateMonitor(logging_interval="step")
    trainer = Trainer(
        max_epochs=epochs, default_root_dir=out_dir, callbacks=[es_callback, lr_logger]
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer


def tune_model(
    datamodule: OrdinancesDataModule,
    encoder_model: str,
    epochs: int,
    dropout_rate: float,
    n_trials: int = 10,
    metric: str = "val_MD@F1",
) -> Tuple[float, Mapping[str, Any]]:
    seed_everything(42)
    # Training and validation data
    tag_to_id = datamodule.tag_to_id
    train_data = datamodule.train_dataloader()
    tune_data = datamodule.tune_dataloader()
    # Number of training and warm-up steps
    num_training_steps, num_warmup_steps = datamodule.num_training_steps(epochs)

    def objective(trial):
        # Extracts hyperparameters
        lr = trial.suggest_float("lr", low=1e-5, high=5e-4)
        weight_decay = trial.suggest_float("weight_decay", low=0.01, high=0.1)
        dropout_rate = trial.suggest_float("dropout_rate", low=0.01, high=0.1)
        grad_norm = trial.suggest_float("grad_norm", low=1.0, high=5.0)
        hyperparameters = {
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "grad_norm": grad_norm,
        }
        # Creates the model
        model = NERBaseAnnotator(
            encoder_model,
            tag_to_id,
            lr,
            num_training_steps,
            num_warmup_steps,
            dropout_rate,
            stage="tuning",
            weight_decay=weight_decay,
        )
        # Creates the trainer
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=metric)
        lr_logger = LearningRateMonitor(logging_interval="step")
        trainer = Trainer(
            max_epochs=epochs,
            enable_checkpointing=False,
            callbacks=[pruning_callback, lr_logger],
            gradient_clip_algorithm="norm",
            gradient_clip_val=grad_norm,
        )
        trainer.logger.log_hyperparams(hyperparameters)
        # Fits the model
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=tune_data)
        # Returns the best metric
        return trainer.logged_metrics["val_MD@F1"].item()

    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_value, study.best_params


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
