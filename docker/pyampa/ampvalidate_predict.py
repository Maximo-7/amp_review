import argparse
import pickle
import re
import pandas as pd
from Bio import SeqIO
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_sequence(sequence):
    """Split a peptide sequence into overlapping dipeptides."""
    return ' '.join([sequence[i:i+2] for i in range(len(sequence) - 1)])


# ── FASTA reader ──────────────────────────────────────────────────────────────

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

def predict(fasta_file, model_path, vectorizer_path, output_csv):

    # Load AMPValidate model and vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Read sequences
    df = fasta_to_df(fasta_file)
    sequences = df["Sequence"].tolist()

    # Preprocess and predict
    sequences_split = [split_sequence(seq) for seq in sequences]
    X = pd.DataFrame(
        vectorizer.transform(sequences_split).toarray(),
        columns=vectorizer.get_feature_names_out()
    )   
    probs = model.predict_proba(X)[:, 1]

    # Build results
    df["Prob_AMP"] = probs
    df["Class"] = ["AMP" if p >= 0.5 else "non-AMP" for p in probs]
    df = df[["ID", "Sequence", "Prob_AMP", "Class"]]

    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMPValidate prediction script")
    parser.add_argument("--fasta",       required=True,              help="Input FASTA file")
    parser.add_argument("--model",       default="/app/pyampa/models/pyampa/AMPValidate.pkl",  help="Path to AMPValidate model (.pkl)")
    parser.add_argument("--vectorizer",  default="/app/pyampa/models/pyampa/amp_validate_vectorizer.pkl", help="Path to AMPValidate vectorizer (.pkl)")
    parser.add_argument("--output",      default="ampvalidate_predictions.csv",     help="Output CSV file")
    args = parser.parse_args()

    predict(args.fasta, args.model, args.vectorizer, args.output)
