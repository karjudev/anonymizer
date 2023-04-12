from typing import List, Mapping, Tuple


def get_mappings(
    records: List[Mapping[str, int | str]],
) -> Tuple[Mapping[str, int], Mapping[int, str]]:
    """Computes the mappings from label to ID and from ID to label.

    :param records: Records to scan.
    :type records: List[Mapping[str, int | str]]
    :return: Unique number of labels, mappings from labels to IDs and from IDs to labels
    :rtype: Tuple[Mapping[str, int], Mapping[int, str]]
    """
    # Unique labels in the dataset
    unique_labels = set(
        span["label"] for sentence in records for span in sentence["entities"]
    )
    # Set of prefixes for the IOB schema
    prefixes = ["B", "I"]
    # Mappings
    label2id = {"O": 0}
    id2label = {0: "O"}
    i = 1
    for label in sorted(unique_labels):
        for prefix in prefixes:
            iob_label = f"{prefix}-{label}"
            label2id[iob_label] = i
            id2label[i] = iob_label
            i += 1
    return label2id, id2label


def prodigy_to_labels(
    spans: List[Mapping[str, int | str]],
    offsets: List[Tuple[int, int]],
    label2id: Mapping[str, int],
) -> List[int]:
    """Converts a list of Prodigy-style charachter-encoded spans to a list of IOB labels.

    :param spans: List of char-indexed spans in the form `{"start": ..., "end": ..., "label": ...}`. The spans are not assumed to be sorted.
    :type spans: List[Mapping[str, int | str]]
    :param offsets: For each token, its char mapping
    :type offsets: List[Tuple[int, int]]
    :param label2id: For each string label, its numerical ID.
    :type label2id: Mapping[str, int]
    :return: List of integer IDs.
    :rtype: List[int]
    """
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
    offsets: List[Tuple[int, int]],
    out_label: str = "O",
) -> List[Mapping[str, int | str]]:
    prodigy_spans = []
    for (start, end), label in spans.items():
        if label != out_label:
            char_start = offsets[start][0].item()
            char_end = offsets[end][1].item()
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
