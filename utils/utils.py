import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Tuple

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import optuna
from optuna.pruners import MedianPruner
from data.ordinances import BatchDataModule, IncrementalDataModule, NERDataModule

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


def load_model(
    model_file: str,
    eval_label2id: Mapping[str, int],
    training_label2id: Mapping[str, int],
    stage="prediction",
):
    if ~os.path.isfile(model_file):
        model_file = get_models_for_evaluation(model_file)

    hparams_file = model_file[: model_file.rindex("checkpoints/")] + "/hparams.yaml"
    model = NERBaseAnnotator.load_from_checkpoint(
        model_file,
        hparams_file=hparams_file,
        eval_label2id=eval_label2id,
        training_label2id=training_label2id,
        stage=stage,
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


def train_incremental(
    model: NERBaseAnnotator,
    datamodule: IncrementalDataModule,
    epochs: int,
    out_dir: Path,
    grad_norm: float,
) -> Trainer:
    return train_model(
        model,
        datamodule,
        epochs,
        out_dir,
        grad_norm,
        reload_every_epoch=True,
    )


def tune_model(
    datamodule: NERDataModule,
    encoder_model: str,
    epochs: int,
    n_trials: int,
    incremental: bool,
    metric: str,
) -> Tuple[float, Mapping[str, Any]]:
    seed_everything(42)
    # Number of training and warm-up steps
    num_training_steps, num_warmup_steps = datamodule.training_warmup_steps(epochs)

    def objective(trial: optuna.Trial):
        # Extracts hyperparameters
        lr = trial.suggest_float("lr", low=1e-5, high=5e-4)
        weight_decay = trial.suggest_float("weight_decay", low=0.001, high=0.01)
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
            encoder_model=encoder_model,
            eval_label2id=datamodule.eval_label2id,
            training_label2id=datamodule.training_label2id,
            lr=lr,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            stage="tuning",
        )
        # Creates the trainer
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=metric)
        lr_logger = LearningRateMonitor(logging_interval="step")
        # Distinguishes between incremental and batch training
        if incremental:
            log_every_n_steps = datamodule.batch_size
            reload_dataloaders_every_n_epochs = 1
        else:
            log_every_n_steps = 50
            reload_dataloaders_every_n_epochs = 0
        trainer = Trainer(
            max_epochs=epochs,
            enable_checkpointing=False,
            callbacks=[pruning_callback, lr_logger],
            gradient_clip_algorithm="norm",
            gradient_clip_val=grad_norm,
            log_every_n_steps=log_every_n_steps,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        )
        trainer.logger.log_hyperparams(hyperparameters)
        # Fits the model
        datamodule.reset()
        trainer.fit(model, datamodule=datamodule)
        # Returns the best metric
        return trainer.logged_metrics[metric].item()

    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_value, study.best_params


def tune_batch(
    datamodule: BatchDataModule,
    encoder_model: str,
    epochs: int,
    n_trials: int = 10,
    metric: str = "val_MD@F1",
) -> Tuple[float, Mapping[str, Any]]:
    return tune_model(
        datamodule,
        encoder_model,
        epochs,
        n_trials=n_trials,
        incremental=False,
        metric=metric,
    )


def tune_incremental(
    datamodule: BatchDataModule,
    encoder_model: str,
    epochs: int = 16,
    n_trials: int = 10,
    metric: str = "MD@F1",
) -> Tuple[float, Mapping[str, Any]]:
    return tune_model(
        datamodule,
        encoder_model,
        epochs,
        n_trials=n_trials,
        incremental=True,
        metric=metric,
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
