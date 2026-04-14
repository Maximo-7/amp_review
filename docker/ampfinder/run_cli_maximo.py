"""
run_cli_maximo.py
Command-line interface for AMPFinder AMP identification, bypassing the GUI.
Replicates predict_amp() from main_windows.py without any Qt dependency.

Source: https://github.com/abcair/AMPFinder
Fix applied: mat[x][y] instead of mat[x+y] (blosum package version compatibility)

Usage:
    python run_cli_maximo.py \
        --input  evaluation_dataset.fasta \
        --output ampfinder_predictions.txt \
        --threshold 0.5
"""

import argparse
import os

import blosum as bl
import joblib
import numpy as np
from Bio import SeqIO
from protlearn import features
from propy import PyPro

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

std = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K",
       "M", "F", "P", "S", "T", "W", "Y", "V"]


def get_blosum_80(seq):
    mat = bl.BLOSUM(80)
    res = np.zeros((20, 20))
    i = 0
    while i + 1 < len(seq):
        x = seq[i]
        y = seq[i + 1]
        val = mat[x][y]   # fix: was mat[x + y]
        k = std.index(x)
        m = std.index(y)
        res[k, m] = val
        i += 1
    return res.flatten().tolist()


def get_ctd(seq):
    return list(PyPro.GetProDes(seq).GetCTD().values())


def get_dpc(seq):
    return list(PyPro.GetProDes(seq).GetDPComp().values())


def get_aaindex1(seq):
    return list(features.aaindex1(seq)[0].flatten())


def pred_encoding(seq):
    vec = get_ctd(seq) + get_dpc(seq) + get_aaindex1(seq) + get_blosum_80(seq)
    return np.array(vec)


def predict_amp(fasta_path, rf_model, threshold):
    results = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = str(record.id)
        seq    = str(record.seq).upper()
        seq    = "".join([aa for aa in seq if aa in std])
        if not seq:
            continue
        vec  = pred_encoding(seq)
        prob = rf_model.predict_proba(vec.reshape(1, -1))[:, 1][0]
        label = 1 if prob > threshold else 0
        results.append((seq_id, seq, prob, label))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="AMPFinder CLI - AMP identification without GUI."
    )
    parser.add_argument("--input",     required=True,
                        help="Path to input FASTA file.")
    parser.add_argument("--output",    required=True,
                        help="Path to output CSV file.")
    parser.add_argument("--model",     default="/app/ampfinder/models/ampfinder/AMPFinder.identify.rf",
                        help="Path to the AMPFinder RF model.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold (default: 0.5).")
    return parser.parse_args()


def main():
    args = parse_args()

    rf_model = joblib.load(args.model)

    results = predict_amp(args.input, rf_model, args.threshold)

    with open(args.output, "w") as f:
        f.write("ID,Sequence,AMPFinder_pred_prob,AMPFinder_pred_label\n")
        for seq_id, seq, prob, label in results:
            f.write(f"{seq_id},{seq},{prob},{label}\n")

    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()