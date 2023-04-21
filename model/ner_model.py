from typing import Dict, List, Literal, Mapping, Optional, Tuple
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
from data.spans import extract_spans, merge_labels
from log import logger

from utils.metric import SpanF1


class NERBaseAnnotator(pl.LightningModule):
    def __init__(
        self,
        encoder_model: str,
        label2id_full: Dict[str, int],
        label2id_filtered: Dict[str, int],
        lr: float,
        num_training_steps: int,
        num_warmup_steps: int,
        dropout_rate: float = 0.1,
        weight_decay: float = 0.01,
        stage: Literal["training", "prediction"] = "training",
    ):
        super(NERBaseAnnotator, self).__init__()

        self.id2label_full = {v: k for k, v in label2id_full.items()}
        self.label2id_full = label2id_full
        self.id2label_filtered = {v: k for k, v in label2id_filtered.items()}
        self.label2id_filtered = label2id_filtered

        self.stage = stage
        target_size = len(self.id2label_filtered)
        logger.info(f"All the Labels:\t{self.id2label_full}")
        logger.info(f"Filtered Labels:\t{self.id2label_filtered}")

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)

        self.feedforward = nn.Linear(
            in_features=self.encoder.config.hidden_size, out_features=target_size
        )

        self.crf_layer = ConditionalRandomField(
            num_tags=target_size,
            constraints=allowed_transitions(
                constraint_type="BIO", labels=self.id2label_filtered
            ),
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.span_f1 = SpanF1()
        self.save_hyperparameters(ignore=["label2id_full", "label2id_filtered"])
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heuristics: List[List[int]],
        spans: Optional[Dict[Tuple[int, int], str]] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = "prediction",
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)

        embedded_text_input = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)
        token_scores = F.log_softmax(token_scores, dim=-1)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(
            token_scores=token_scores,
            attention_mask=attention_mask,
            heuristics=heuristics,
            labels=labels,
            spans=spans,
            batch_size=batch_size,
            mode=mode,
        )
        return output

    def perform_forward_step(self, batch, mode=""):
        input_ids, attention_mask, heuristics, spans, labels = batch
        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            heuristics=heuristics,
            spans=spans,
            labels=labels,
            mode=mode,
        )
        return output

    def _compute_token_tags(
        self,
        token_scores: torch.Tensor,
        attention_mask: torch.Tensor,
        heuristics: List[List[int]],
        labels: torch.Tensor,
        spans: Optional[Mapping[Tuple[int, int], str]],
        batch_size,
        mode="",
    ):
        best_path = self.crf_layer.viterbi_tags(token_scores, attention_mask)
        pred_results = []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            tag_seq = merge_labels(
                tag_seq, heuristics[i], out_idx=self.label2id_full["O"]
            )
            pred_results.append(extract_spans(tag_seq, self.id2label_full))
        output = dict()
        if mode == "prediction":
            output["tags"] = pred_results
        if labels is not None:
            loss = -self.crf_layer(token_scores, labels, attention_mask) / float(
                batch_size
            )
            self.span_f1(pred_results, spans)
            output["loss"] = loss
            output["results"] = self.span_f1.get_metric()
        return output

    def predict_tags(self, batch, device="cuda:0"):
        input_ids, attention_mask, heuristics, spans, labels = batch
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )
        batch = input_ids, attention_mask, heuristics, spans, labels

        pred_tags = self.perform_forward_step(batch, mode="prediction")["tags"]
        return pred_tags
