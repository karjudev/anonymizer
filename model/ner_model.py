from typing import Dict, List, Literal, Mapping, Optional, Tuple

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
from data.ordinances import Example
from data.spans import extract_spans
from log import logger

from utils.metric import SpanF1


class NERBaseAnnotator(pl.LightningModule):
    def __init__(
        self,
        encoder_model: str,
        label2id: Dict[str, int],
        lr: float,
        num_training_steps: int,
        num_warmup_steps: int,
        dropout_rate: float = 0.1,
        weight_decay: float = 0.01,
        stage: Literal["training", "prediction"] = "training",
    ):
        super(NERBaseAnnotator, self).__init__()

        self.label2id = label2id
        self.id2label = {idx: label for label, idx in label2id.items()}

        self.stage = stage
        target_size = len(self.id2label)

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)

        self.feedforward = nn.Linear(
            in_features=self.encoder.config.hidden_size, out_features=target_size
        )

        self.crf_layer = ConditionalRandomField(
            num_tags=target_size,
            constraints=allowed_transitions(
                constraint_type="BIO", labels=self.id2label
            ),
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.span_f1 = SpanF1()
        self.save_hyperparameters(ignore=["label2id", "id2label"])
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

    def __unwrap_batch(self, batch: Example) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch[0], torch.Tensor):
            input_ids, attention_mask = batch
        else:
            input_ids, attention_mask = batch[0]
        return input_ids, attention_mask

    def forward(self, batch: Example) -> torch.Tensor:
        input_ids, attention_mask = self.__unwrap_batch(batch)

        embedded_text_input = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)
        token_scores = F.log_softmax(token_scores, dim=-1)

        return token_scores

    def embed(self, batch):
        input_ids, attention_mask = self.__unwrap_batch(batch)
        embedded_text_input = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_token = embedded_text_input.last_hidden_state[:, 0]
        return cls_token

    def common_step(self, batch):
        if len(batch) == 2:
            (input_ids, attention_mask), labels = batch
        else:
            (input_ids, attention_mask), labels, _ = batch
        predictions = self(batch)
        output = self._compute_token_tags(
            predictions, attention_mask, labels, batch_size=input_ids.size(0)
        )
        return output

    def _compute_token_tags(
        self,
        token_scores: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
    ):
        # Extracts spans from true labels
        spans = [extract_spans(lab, self.id2label) for lab in labels.tolist()]
        # Extracts spans from predicted tokens
        best_path = self.crf_layer.viterbi_tags(token_scores, attention_mask)
        pred_results = []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_results.append(extract_spans(tag_seq, self.id2label))
        # Evaluates the model
        self.span_f1(pred_results, spans)
        # Computes the loss
        loss = -self.crf_layer(token_scores, labels, attention_mask) / float(batch_size)
        # Outputs scores, loss and results
        output = dict()
        output["loss"] = loss
        output["results"] = self.span_f1.get_metric()
        return output

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log_metrics(
            output["results"],
            loss=output["loss"],
            suffix="",
            on_step=True,
            on_epoch=False,
        )
        self.training_outputs.append(output)
        return output

    def on_training_epoch_end(self) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds["loss"].item() for preds in self.training_outputs])
        self.log_metrics(
            pred_results, loss=avg_loss, suffix="", on_step=False, on_epoch=True
        )
        self.training_outputs.clear()

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log_metrics(
            output["results"],
            loss=output["loss"],
            suffix="val_",
            on_step=True,
            on_epoch=False,
        )
        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        pred_results = self.span_f1.get_metric(True)
        avg_loss = np.mean([preds["loss"].item() for preds in self.validation_outputs])
        self.log_metrics(
            pred_results, loss=avg_loss, suffix="val_", on_step=False, on_epoch=True
        )
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log_metrics(
            output["results"],
            loss=output["loss"],
            suffix="_t",
            on_step=True,
            on_epoch=False,
        )
        self.testing_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        pred_results = self.span_f1.get_metric()
        avg_loss = np.mean([preds["loss"].item() for preds in self.testing_outputs])
        self.log_metrics(pred_results, loss=avg_loss, on_step=False, on_epoch=True)
        out = {"test_loss": avg_loss, "results": pred_results}
        self.testing_outputs.clear()
        return out

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
