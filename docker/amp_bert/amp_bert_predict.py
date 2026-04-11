# Inference code by Alvaro Maximo.
# Predicts AMP probability for peptide sequences using a fine-tuned AMP-BERT model.
# Source (training code): https://github.com/GIST-CSBL/AMP-BERT

import argparse
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification


def preprocess_sequence(seq):
    spaced_seq = " ".join(seq)
    spaced_seq = re.sub(r"[UZOB]", "X", spaced_seq)
    return spaced_seq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict AMPs using a fine-tuned AMP-BERT model."
    )
    parser.add_argument("--input_csv",  required=True,
                        help="CSV with 'ID' and 'Sequence' columns.")
    parser.add_argument("--model_dir",  required=True,
                        help="Directory containing the fine-tuned model.")
    parser.add_argument("--output_csv", required=True,
                        help="Output CSV with predictions.")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    tokenizer = AutoTokenizer.from_pretrained(
        "Rostlab/prot_bert_bfd", do_lower_case=False
    )
    model = BertForSequenceClassification.from_pretrained(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    model.eval()

    df["processed_seq"] = df["Sequence"].apply(preprocess_sequence)

    ids         = []
    sequences   = []
    predictions = []
    probs       = []

    with torch.no_grad():
        for i in tqdm(range(0, len(df), args.batch_size), desc="Processing batches"):
            batch_df   = df.iloc[i:i + args.batch_size]
            batch_seqs = batch_df["processed_seq"].tolist()
            batch_ids  = batch_df["ID"].tolist()
            batch_orig = batch_df["Sequence"].tolist()

            batch_tok = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            batch_tok = {k: v.to(device) for k, v in batch_tok.items()}

            output      = model(**batch_tok)
            batch_probs = torch.sigmoid(output.logits)[:, 1].detach().cpu().numpy()
            batch_preds = batch_probs > 0.5

            ids.extend(batch_ids)
            sequences.extend(batch_orig)
            predictions.extend([1 if pred else 0 for pred in batch_preds])
            probs.extend(batch_probs)

    results = pd.DataFrame({
        "ID":                  ids,
        "Sequence":            sequences,
        "AMP-BERT_pred_label": predictions,
        "AMP-BERT_pred_prob":  probs,
    })

    results.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
