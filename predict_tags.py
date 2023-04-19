from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from utils.utils import get_reader, load_model, get_out_filename, get_tagset


def main(
    test: Path,
    model: Path,
    out_dir: Path,
    prefix: str,
    iob_tagging: str = "kind",
    batch_size: int = 128,
    max_instances: int = -1,
    max_length: int = 512,
    cuda: str = "cuda:0",
) -> None:
    # load the dataset first
    test_data = get_reader(
        file_path=test,
        target_vocab=get_tagset(iob_tagging),
        max_instances=max_instances,
        max_length=max_length,
    )

    model, model_file = load_model(model, label2id=get_tagset(iob_tagging))
    model = model.to(cuda)
    # use pytorch lightnings saver here.
    eval_file = get_out_filename(out_dir, model_file, prefix=prefix)

    test_dataloaders = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=model.collate_batch,
        shuffle=False,
        drop_last=False,
    )
    out_str = ""
    index = 0
    for batch in tqdm(test_dataloaders, total=len(test_dataloaders)):
        pred_tags = model.predict_tags(batch, device=cuda)

        for pred_tag_inst in pred_tags:
            out_str += "\n".join(pred_tag_inst)
            out_str += "\n\n\n"
        index += 1
    open(eval_file, "wt").write(out_str)


if __name__ == "__main__":
    typer.run(main)
