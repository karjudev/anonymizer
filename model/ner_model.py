from typing import Dict, Literal
from itertools import compress

import lightning.pytorch as pl

import torch
import torch.nn.functional as F
import numpy as np

from allennlp_light import ConditionalRandomField
from allennlp_light.modules.conditional_random_field.conditional_random_field import (
    allowed_transitions,
)
from torch import nn
from transformers import get_linear_schedule_with_warmup, AutoModel
from data.spans import extract_spans

from utils.metric import SpanF1


class NERBaseAnnotator(pl.LightningModule):
    def __init__(
        self,
        encoder_model: str,
        tag_to_id: Dict[str, int],
        lr: float,
        num_training_steps: int,
        num_warmup_steps: int,
        dropout_rate: float = 0.1,
        weight_decay: float = 0.01,
        stage: Literal["training", "evaluation", "prediction"] = "training",
    ):
        super(NERBaseAnnotator, self).__init__()

        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id

        self.stage = stage
        target_size = len(self.id_to_tag)

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)

        self.feedforward = nn.Linear(
            in_features=self.encoder.config.hidden_size, out_features=target_size
        )

        self.crf_layer = ConditionalRandomField(
            num_tags=target_size,
            constraints=allowed_transitions(
                constraint_type="BIO", labels=self.id_to_tag
            ),
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.span_f1 = SpanF1()
        self.save_hyperparameters(ignore="tag_to_id")
        self.training_outputs = []
        self.validation_outputs = []
        self.testing_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.stage == "training":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.hparams.num_training_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        return [optimizer]

    def on_test_epoch_end(self):
        pred_results = self.span_f1.get_metric()
        avg_loss = np.mean([preds["loss"].item() for preds in self.testing_outputs])
        self.log_metrics(pred_results, loss=avg_loss, on_step=False, on_epoch=True)
        out = {"test_loss": avg_loss, "results": pred_results}
        self.testing_outputs.clear()
        return out

    def on_training_epoch_end(self) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds["loss"].item() for preds in self.training_outputs])
        self.log_metrics(
            pred_results, loss=avg_loss, suffix="", on_step=False, on_epoch=True
        )
        self.training_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds["loss"].item() for preds in self.validation_outputs])
        self.log_metrics(
            pred_results, loss=avg_loss, suffix="val_", on_step=False, on_epoch=True
        )
        self.validation_outputs.clear()

    def validation_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch)
        self.log_metrics(
            output["results"],
            loss=output["loss"],
            suffix="val_",
            on_step=True,
            on_epoch=False,
        )
        self.validation_outputs.append(output)
        return output

    def training_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch)
        self.log_metrics(
            output["results"],
            loss=output["loss"],
            suffix="",
            on_step=True,
            on_epoch=False,
        )
        self.training_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        output = self.perform_forward_step(batch, mode=self.stage)
        self.log_metrics(
            output["results"],
            loss=output["loss"],
            suffix="_t",
            on_step=True,
            on_epoch=False,
        )
        self.testing_outputs.append(output)
        return output

    def log_metrics(
        self, pred_results, loss=0.0, suffix="", on_step=False, on_epoch=True
    ):
        for key in pred_results:
            self.log(
                suffix + key,
                pred_results[key],
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
                logger=True,
            )

        self.log(
            suffix + "loss",
            loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            logger=True,
        )

    def perform_forward_step(self, batch, mode=""):
        tokens, tags, mask, token_mask, metadata = batch
        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)
        token_scores = F.log_softmax(token_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(
            token_scores=token_scores,
            mask=mask,
            tags=tags,
            metadata=metadata,
            batch_size=batch_size,
            mode=mode,
        )
        return output

    def _compute_token_tags(
        self, token_scores, mask, tags, metadata, batch_size, mode=""
    ):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf_layer(token_scores, tags, mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, mask)

        pred_results, pred_tags = [], []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_tags.append([self.id_to_tag[x] for x in tag_seq])
            pred_results.append(extract_spans(tag_seq, self.id_to_tag))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}

        if mode == "predict":
            output["token_tags"] = pred_tags
        return output

    def predict_tags(self, batch, device="cuda:0"):
        tokens, tags, mask, token_mask, metadata = batch
        tokens, mask, token_mask, tags = (
            tokens.to(device),
            mask.to(device),
            token_mask.to(device),
            tags.to(device),
        )
        batch = tokens, tags, mask, token_mask, metadata

        pred_tags = self.perform_forward_step(batch, mode="predict")["token_tags"]
        tag_results = [
            compress(pred_tags_, mask_)
            for pred_tags_, mask_ in zip(pred_tags, token_mask)
        ]
        return tag_results
