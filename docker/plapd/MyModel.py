# Adapted from MyModel.py
# Source: https://github.com/lichaozhang2/PLAPD
# Changes: ─────── Added ─────── blocks and # added or # modified lines

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import esm
import math
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# ─────── Added ───────────────────────────────────
from Bio import SeqIO
import pandas as pd
import re
import numpy as np
import argparse

def fasta_to_df(fasta_file):
    """
    Reads a FASTA file and converts it into a dataframe with:
    - ID: first token after '>'
    - Name: the rest of the description
    - Sequence: sequence as string
    """
    records = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description.strip()
        # after ID it comes " " or "|"
        header_parts = re.split(r"[ |]", header, maxsplit = 1)
        seq_ID = header_parts[0] # first token = ID
        name = header_parts[1] if len(header_parts) > 1 else "" # rest = Name
        records.append({"ID": seq_ID, "Name": name, "Sequence": str(record.seq)})

    return pd.DataFrame(records)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict using PLAPD AMP classifier"
    )

    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--output_csv", type=str, default="plapd_predictions.csv", help="Output csv file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()

args = parse_args()
# ─────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('encoding', self._get_timing_signal(max_len, d_model))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

    def _get_timing_signal(self, length, channels):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        pe = torch.zeros(length, channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_len=5000):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, src, tgt):
        src = src + self.pos_encoder(src)
        # src = self.pos_encoder(src)
        output = self.transformer(src, tgt)
        return output


class ESM(nn.Module):
    def __init__(self):
        super(ESM, self).__init__()
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

    def forward(self, prot_seqs):
        data = [('seq{}'.format(i), seq) for i, seq in enumerate(prot_seqs)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        # Extract per-residue representations (on GPU)
        with torch.no_grad():
            results = self.esm_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33][:, 1:-1]

        # [batch, L, 1280]
        prot_embedding = token_representations

        return prot_embedding


class AIMP(torch.nn.Module):
    def __init__(self, pre_feas_dim, hidden, n_transformer, dropout):
        super(AIMP, self).__init__()

        self.esm = ESM()

        self.pre_embedding = nn.Sequential(
            nn.Conv1d(pre_feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.bn = nn.ModuleList([nn.BatchNorm1d(pre_feas_dim)])

        self.n_transformer = n_transformer

        self.transformer = TransformerModel(d_model=hidden, nhead=4, num_layers=self.n_transformer,
                                            dim_feedforward=2048)
        self.transformer_act = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_res = nn.Sequential(
            nn.Conv1d(hidden + hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_pool = nn.AdaptiveAvgPool2d((1, None))
        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.pre_embedding, self.clf]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)
        for layer in [self.transformer_act, self.transformer_res]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)

    def forward(self, protein_sequence):

        bert_output = self.esm(protein_sequence)

        pre_feas = self.bn[0](bert_output.permute(0, 2, 1)).permute(0, 2, 1)

        pre_feas = self.pre_embedding(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)

        transformer_out = self.transformer(pre_feas, pre_feas)
        transformer_out = self.transformer_act(transformer_out.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_out = self.transformer_res(torch.cat([transformer_out, pre_feas], dim=-1).permute(0, 2, 1)).permute(
            0, 2, 1)
        transformer_out = self.transformer_pool(transformer_out).squeeze(1)

        out = self.clf(transformer_out)
        out = torch.nn.functional.softmax(out, -1)
        return out, bert_output, pre_feas, transformer_out
# ─────── Added ───────────────────────────────────
if __name__ == '__main__':
    model = AIMP(pre_feas_dim=1280, hidden=1280, n_transformer=1, dropout=0.5)
    model.cuda()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    df_sequences = fasta_to_df(args.input_fasta)
    df_sequences = df_sequences.drop(columns=["Name"])
# ─────────────────────────────────────────────────
    protein_sequences = df_sequences["Sequence"].tolist() # modified

    # 生成一个形状为[1, 30, 1024]的随机tensor
    # random_tensor = torch.randn(2, 30, 1024)
    batch_size = args.batch_size
    all_probs = []

    for i in range(0, len(protein_sequences), batch_size):
        batch = protein_sequences[i:i+batch_size]
        out, _, _, _ = model(batch)
        probs = out.cpu().detach().numpy()
        all_probs.extend(probs)

    probs = np.array(all_probs)
    prob_amp = probs[:, 1] # added. AMP probability

    labels = ["AMP" if p > 0.5 else "non-AMP" for p in prob_amp]
    df_sequences["Prob_AMP"] = prob_amp # added
    df_sequences["Label"] = labels # added

    df_sequences.to_csv(args.output_csv, index=False) # added
