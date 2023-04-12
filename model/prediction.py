from typing import List, Mapping
from transformers import PreTrainedTokenizer
from data.ordinances import encode_text
from data.spans import spans_to_prodigy
from model.ner_model import NERBaseAnnotator


def predict(
    model: NERBaseAnnotator,
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int = 512,
) -> List[Mapping[str, int | str]]:
    prodigy_spans = []
    for input_ids, attention_mask, offsets, _ in encode_text(
        text, tokenizer, max_length
    ):
        spans = model(
            input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0)
        )["tags"][0]
        prodigy_spans.extend(spans_to_prodigy(spans, offsets))
    return prodigy_spans


def redact(text: str, prodigy_spans: List[Mapping[str, int | str]]) -> str:
    prodigy_spans.sort(key=lambda span: span["start"], reverse=True)
    for span in prodigy_spans:
        text = text[: span["start"]] + span["label"] + text[span["end"] :]
    return text
