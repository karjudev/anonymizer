from typing import List, Mapping, Optional
import re


DATE_REGEX: re.Pattern = re.compile(
    r"\d{1,2}([\/\.-]\d{1,2}[\/\.-]|\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+)\d{2,4}",
    flags=re.MULTILINE | re.IGNORECASE,
)

BORN_REGEX: re.Pattern = re.compile(r"\bnat[oa]\b", flags=re.MULTILINE | re.IGNORECASE)


def detect_dates(
    text: str, binarize: bool, window: Optional[int] = None
) -> List[Mapping[str, int | str]]:
    label = "OMISSIS" if binarize else "TIME"
    spans = []
    for match in DATE_REGEX.finditer(text):
        start, end = match.span()
        if window is not None:
            right = start
            left = max(0, right - window)
            if BORN_REGEX.search(text[left:right]) is not None:
                spans.append({"start": start, "end": end, "label": label})
        else:
            spans.append({"start": start, "end": end, "label": label})
    return spans
