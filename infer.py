import argparse

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from utils import CHECKPOINTS_DIR


def preprocess(text: str, tokenizer: BertTokenizer) -> dict:
    """
    Preprocesses the input text using the provided BERT tokenizer.

    Args:
        text (str): The input text to be tokenized.
        tokenizer (BertTokenizer): The BERT tokenizer to use for encoding the
            text.

    Returns:
        dict: A dictionary containing the tokenized input IDs and attention
            mask.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


def predict_sentiment(
    text: str, model: BertForSequenceClassification, tokenizer: BertTokenizer
):
    """
    Predict the sentiment of a given text using a BERT model.
    Args:
        text (str): The input text to analyze.
        model (BertForSequenceClassification): The pre-trained BERT model for
            sequence classification.
        tokenizer (BertTokenizer): The tokenizer associated with the BERT
            model.
    Returns:
        str: "Positive" if the sentiment is positive, otherwise "Negative".
    """
    input_ids, attention_mask = preprocess(text, tokenizer)

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    prediction = torch.argmax(logits, dim=1).cpu().item()

    return "Positive" if prediction == 1 else "Negative"


def prepare_model_and_tokenizer(
    run_name: str,
) -> tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Load the BERT model and tokenizer from the specified run directory.

    Args:
        run_name (str): The name of the run directory containing the model
            and tokenizer files.

    Returns:
        tuple[BertForSequenceClassification, BertTokenizer]: A tuple containing
            the BERT model and tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForSequenceClassification.from_pretrained(
        CHECKPOINTS_DIR / run_name / "best"
    )
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(CHECKPOINTS_DIR / run_name / "best")
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment analysis inference.")
    parser.add_argument(
        "--run_name",
        type=str,
        default="run",
        help="The name of the run directory containing the model and tokenizer files.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="The input text to analyze.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_name = args.run_name
    if not (CHECKPOINTS_DIR / run_name).exists():
        print(f"Run directory '{run_name}' does not exist.")
        return

    model, tokenizer = prepare_model_and_tokenizer(run_name)

    sentiment = predict_sentiment(args.text, model, tokenizer)

    print(
        f'\nModel predicted that the sentiment of the message: \n\n\t"{args.text}" \nis \n\t"{sentiment}"'
    )


if __name__ == "__main__":
    main()
    # python infer.py --run_name 2024-09-22_22-31-42 --text "I hate this product!"
