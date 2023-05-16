# NER-Based Contextual Anonymization of Legal Documents

This code repository provides the code to train and evaluate a model for contextual text anonymization of legal documents. It provides utilities for:
- Custom data readers
- Usage of any HuggingFace pre-trained transformer models
- Usage of heuristics to detect dates.
- Training and Testing through Pytorch-Lightning

To get a full explanation of all the possible operations you can do with this code, type the following in your terminal:
- Preprocess a JSONL file: `python ./preprocessing/ordinances.py --help`
- Tune a model: `python ./tune_model.py --help`
- Train a model: `python ./train_model.py --help`
- Evaluate a trained model: `python ./evaluate.py --help`

## Setting up the code environment

```bash
$ pip install -r requirements.txt
```

## License 
The code under this repository is licensed under the [Apache 2.0 License](https://github.com/amzn/multiconer-baseline/blob/main/LICENSE).
