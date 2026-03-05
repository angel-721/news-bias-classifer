import string
import sys

import torch
from transformers import DistilBertTokenizer

from config import LABELS, STOPWORDS
from network import NetworkClassifer

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def is_content_token(tok: str) -> bool:
    return (
        tok not in ["[CLS]", "[SEP]", "[PAD]"]
        and not tok.startswith("##")
        and tok not in string.punctuation
        and tok.lower() not in STOPWORDS
        and len(tok) > 2
    )


def predict(text: str, model: NetworkClassifer) -> tuple[str, dict[str, float]]:
    encoding = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, mask)
        probs = torch.softmax(logits, dim=1).squeeze()

    confidence = {LABELS[i]: round(probs[i].item(), 4) for i in range(2)}
    label = max(confidence, key=confidence.get)
    return label, confidence


# Pharses that are the strongest when it comes to the model's label prediction
def get_signal_phrases(
    text: str,
    model: NetworkClassifer,
    top_k: int = 8,
    context_words: int = 12,
) -> list[tuple[str, float]]:
    encoding = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.bert(input_ids=input_ids, attention_mask=mask)
        x = outputs.last_hidden_state
        scores = model.at(x)
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).squeeze()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    real_len = mask[0].sum().item()

    # Only content tokens as anchor candidates
    token_weights = [
        (i, tokens[i], weights[i].item())
        for i in range(1, real_len - 1)
        if is_content_token(tokens[i])
    ]

    token_weights.sort(key=lambda x: -x[2])
    top_tokens = token_weights[:top_k]

    # Build phrases around top anchor tokens, skip overlapping windows
    phrases = []
    seen_positions = set()

    for pos, tok, weight in top_tokens:
        if pos in seen_positions:
            continue

        start = max(1, pos - context_words)
        end = min(real_len - 1, pos + context_words + 1)

        context_tokens = tokens[start:end]
        phrase = tokenizer.convert_tokens_to_string(context_tokens).strip()

        phrases.append((phrase, round(weight, 4)))
        seen_positions.update(range(start, end))

    return phrases


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <path_to_article.txt>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        text = f.read().replace("\n", " ").strip()

    if not text:
        print("Input file is empty.")
        sys.exit(1)

    model = NetworkClassifer().to(device)
    model.load_state_dict(
        torch.load("./models/best_distilbert.pth", map_location=device)
    )
    model.eval()

    label, confidence = predict(text, model)
    print(f"Prediction : {label}")
    print(f"Likely     : {confidence['Likely']:.2%}")
    print(f"Unlikely   : {confidence['Unlikely']:.2%}")

    phrases = get_signal_phrases(text, model)
    if phrases:
        print("\nTop signal phrases:")
        for phrase, weight in phrases:
            print(f"  {weight:.4f}  {phrase!r}")
