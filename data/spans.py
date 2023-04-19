from typing import List, Mapping, Set, Tuple

import torch


MULTICLASS_LABEL2ID = {
    "O": 0,
    "B-LOC": 1,
    "I-LOC": 2,
    "B-MISC": 3,
    "I-MISC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-PER": 7,
    "I-PER": 8,
    "B-TIME": 9,
    "I-TIME": 10,
}


BINARIZED_LABEL2ID = {"O": 0, "B-OMISSIS": 1, "I-OMISSIS": 2}


def get_label2id(binarize: bool, ignore_tags: Set[str] = None) -> Mapping[str, int]:
    if binarize:
        return BINARIZED_LABEL2ID
    label2id = MULTICLASS_LABEL2ID
    if ignore_tags is not None:
        label2id = {
            key: i
            for i, key in enumerate(
                key for key in label2id.keys() if key[2:] not in ignore_tags
            )
        }
    return label2id


def prodigy_to_labels(
    spans: List[Mapping[str, int | str]],
    offsets: List[Tuple[int, int]],
    label2id: Mapping[str, int],
) -> List[int]:
    label_ids = [label2id["O"]] * len(offsets)
    for span in spans:
        i = 1
        while i < len(offsets) - 1 and offsets[i][0] < span["start"]:
            i += 1
        # If we reached the last offset we have to continue with the next span
        if i == len(offsets) - 1:
            continue
        # The target label is reported only in multiclass setting
        label = span["label"]
        # Assigns the "B"-label
        label_ids[i] = label2id[f"B-{label}"]
        i += 1
        # Assigns the "I"-labels
        while i < len(offsets) - 1 and offsets[i][1] <= span["end"]:
            label_ids[i] = label2id[f"I-{label}"]
            i += 1
    return label_ids


def spans_to_prodigy(
    spans: Mapping[Tuple[int, int], str],
    offsets: List[Tuple[int | torch.Tensor, int | torch.Tensor]],
    out_label: str = "O",
) -> List[Mapping[str, int | str]]:
    prodigy_spans = []
    for (start, end), label in spans.items():
        if label != out_label:
            char_start = int(offsets[start][0])
            char_end = int(offsets[end][1])
            prodigy_spans.append({"start": char_start, "end": char_end, "label": label})
    return prodigy_spans


def extract_spans(labels: List[int], id2label: Mapping[int, str]):
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[
            (_cur_start, _cur_id - 1)
        ] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    for _id, idx in enumerate(labels):
        nt = id2label[idx]
        indicator = nt[0]
        if indicator == "B":
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == "I":
            # do nothing
            pass
        elif indicator == "O":
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = "O"
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans
