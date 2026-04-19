# Inference code by Alvaro Maximo.
# Predicts AMP probability for peptide sequences using a ProtT5-XL-UniRef50 model.
# Source (training code): https://github.com/liangxiaodata/AMPpred

import pandas as pd
import torch
import re
import numpy as np
import argparse
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel
import torch.nn as nn


# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ── Model definition ─────────────────────────────────────────────────────────
class ImprovedBinaryClassificationMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256, 128, 64, 32], output_dim=2):
        super(ImprovedBinaryClassificationMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Identity):
                x = x + layer(x)
            else:
                x = layer(x)
        return x

class PretrainedAndClassifierModel(nn.Module):
    def __init__(self, pretrained_model):
        super(PretrainedAndClassifierModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier_model = ImprovedBinaryClassificationMLP(
            input_dim=1024, hidden_dims=[512, 256, 128, 64, 32], output_dim=2
        )
        self.average_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids, attention_mask=attention_mask)
        emb_0 = outputs.last_hidden_state
        pooled_sequence = self.average_pooling(emb_0.permute(0, 2, 1)).squeeze(2)
        return self.classifier_model(pooled_sequence)


# ── FASTA reader ─────────────────────────────────────────────────────────────
def fasta_to_df(fasta_file):
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description.strip()
        header_parts = re.split(r"[ |]", header, maxsplit=1)
        seq_id = header_parts[0]
        name = header_parts[1] if len(header_parts) > 1 else ""
        records.append({"ID": seq_id, "Name": name, "Sequence": str(record.seq)})
    return pd.DataFrame(records)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(fasta_file, model_dir, classifier_weights, output_csv, batch_size=8, tokenizer_dir=None):

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir or model_dir, do_lower_case=False)

    encoder = T5EncoderModel.from_pretrained(model_dir)
    model = PretrainedAndClassifierModel(pretrained_model=encoder)
    model.classifier_model.load_state_dict(torch.load(classifier_weights, map_location=device))
    model.to(device)
    model.eval()

    # Read sequences
    df = fasta_to_df(fasta_file)
    sequences = df["Sequence"].tolist()

    all_probs = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]

            # Preprocess
            encoded = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch_seqs]
            ids = tokenizer.batch_encode_plus(encoded, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    # Build results
    df["Prob_AMP"] = all_probs
    df["Class"] = ["AMP" if p >= 0.5 else "non-AMP" for p in all_probs]
    df = df[["ID", "Sequence", "Prob_AMP", "Class"]]

    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KT-AMPpred prediction script")
    parser.add_argument("--fasta",       required=True,  help="Input FASTA file")
    parser.add_argument("--tokenizer_dir", default="/app/kt_amppred/prot_t5_xl_half_uniref50-enc", help="Directory with the original ProtT5 tokenizer")
    parser.add_argument("--model_dir",     default="finetune_peptide_model", help="Directory with the fine-tuned T5 encoder")
    parser.add_argument("--classifier",  default="finetune_peptide_model/classifier_weights.pth", help="Path to classifier weights")
    parser.add_argument("--output",      default="kt_amppred_predictions.csv", help="Output CSV file")
    parser.add_argument("--batch_size",  type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    predict(args.fasta, args.model_dir, args.classifier, args.output, args.batch_size, args.tokenizer_dir)
