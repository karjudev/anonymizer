import os
import time

import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import optuna
from optuna.pruners import MedianPruner

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


def create_model(
    train_data,
    dev_data,
    tag_to_id,
    batch_size=64,
    dropout_rate=0.1,
    stage="fit",
    lr=1e-5,
    encoder_model="xlm-roberta-large",
    num_gpus=1,
):
    return NERBaseAnnotator(
        train_data=train_data,
        dev_data=dev_data,
        tag_to_id=tag_to_id,
        batch_size=batch_size,
        stage=stage,
        encoder_model=encoder_model,
        dropout_rate=dropout_rate,
        lr=lr,
        pad_token_id=train_data.pad_token_id,
        num_gpus=num_gpus,
    )


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


def train_model(model, out_dir=None, epochs=10, gpus=1, trial=None):
    trainer = get_trainer(gpus=gpus, out_dir=out_dir, epochs=epochs, trial=trial)
    trainer.fit(model)
    return trainer


def tune_model(
    train_data,
    tune_data,
    encoder_model,
    tag_to_id,
    batch_size,
    dropout_rate,
    epochs,
    n_trials=10,
):
    def objective(trial):
        lr = trial.suggest_float("lr", low=1e-5, high=5e-4)
        hyperparameters = {"lr": lr}
        model = create_model(
            train_data,
            tune_data,
            tag_to_id,
            batch_size,
            dropout_rate,
            stage="tuning",
            lr=lr,
            encoder_model=encoder_model,
        )
        trainer = train_model(model=model, epochs=epochs, trial=trial)
        trainer.logger.log_hyperparams(hyperparameters)
        return trainer.logged_metrics["val_MD@F1"].item()

    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_value, study.best_params


def get_trainer(gpus=4, is_test=False, out_dir=None, epochs=10, trial=None):
    enable_checkpointing = out_dir is not None
    seed_everything(42)
    if is_test:
        return (
            Trainer(devices=1, enable_checkpointing=enable_checkpointing)
            if torch.cuda.is_available()
            else Trainer(
                val_check_interval=100, enable_checkpointing=enable_checkpointing
            )
        )

    if torch.cuda.is_available():
        trainer = Trainer(
            devices=gpus,
            deterministic=True,
            max_epochs=epochs,
            callbacks=[get_model_earlystopping_callback(trial)],
            default_root_dir=out_dir,
            enable_checkpointing=enable_checkpointing,
        )
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = Trainer(
            max_epochs=epochs,
            enable_checkpointing=enable_checkpointing,
            default_root_dir=out_dir,
        )

    return trainer


def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return lr_monitor


def get_model_earlystopping_callback(trial=None, metric="val_MD@F1"):
    if trial is None:
        es_clb = EarlyStopping(
            monitor=metric, min_delta=0.001, patience=3, verbose=True, mode="min"
        )
    else:
        es_clb = PyTorchLightningPruningCallback(trial, monitor=metric)
    return es_clb


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
