import math
from pathlib import Path
import typer

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from data.ordinances import load_domain_adaptation


def main(
    data_dir: Path,
    output_dir: Path,
    encoder_model: str,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    epochs: int = 2,
    mlm_probability: float = 0.15,
    batch_size: int = 16,
    max_length: int = 512,
    random_seed: int = 0,
) -> None:
    # Loads the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    # Loads the domain adaptation datasets
    datasets = load_domain_adaptation(data_dir, tokenizer, max_length)
    # Data collator for language modelling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    # Number of steps in one epoch
    steps_per_epoch = len(datasets["training"]) // batch_size
    # Arguments for the trainer
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        save_steps=steps_per_epoch,
        save_total_limit=3,
        weight_decay=weight_decay,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=random_seed,
    )
    # Trains the model
    model = AutoModelForMaskedLM.from_pretrained(encoder_model)
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=datasets["training"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir / "final_model")
    # Evaluates the model performances
    eval_results = trainer.evaluate(eval_dataset=datasets["evaluation"])
    print("Evaluation results: ", eval_results)
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.3f}")
    print("----------------\n")


if __name__ == "__main__":
    typer.run(main)
