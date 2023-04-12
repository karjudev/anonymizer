from pathlib import Path
import srsly
import typer


def represent_span(text, span, window) -> str:
    start = span["start"]
    end = span["end"]
    label = span["label"]
    left_text = text[start - window : start]
    span_text = text[start:end]
    right_text = text[end : end + window]
    return f"{left_text}[{span_text}]({label}){right_text}"


def main(path: Path, window: int = 25) -> None:
    for record in srsly.read_jsonl(path):
        text = record["text"]
        ground_truth = record["ground_truth"]
        predicted = record["predicted"]
        fp = 0
        for pred_span in predicted:
            if pred_span not in ground_truth:
                span_text = represent_span(text, pred_span, window)
                typer.echo("False positive")
                typer.echo(span_text)
                fp += 1
                input(">")
        typer.echo(f"False positives: {fp} / {len(predicted)}")
        fn = 0
        for true_span in ground_truth:
            if true_span not in predicted:
                span_text = represent_span(text, true_span, window)
                typer.echo("False negative (missed entity)")
                typer.echo(span_text)
                fn += 1
                input(">")
        typer.echo(f"False negatives: {fn} / {len(ground_truth)}")


if __name__ == "__main__":
    typer.run(main)
