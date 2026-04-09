# Adapted from 3_Testing_Models.py
# Source: https://github.com/williamdee1/LMPred_AMP_Prediction
# Changes: new script for CLI prediction on evaluation dataset.
# The model with the best reported performance (T5 trained on UniRef50) was selected.

import argparse
import numpy as np
import pandas as pd
from tensorflow import keras


# ─── Helper (original) ────────────────────────────────────────────────────────

def convert_preds(preds):
    model_preds = np.concatenate(np.round(preds), axis=0).tolist()
    df = pd.DataFrame({'pred_labels': model_preds})
    pred_labels = df.pred_labels
    return pred_labels


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict AMPs using a trained LMPred model."
    )
    parser.add_argument("--embeddings_npy", required=True,
                        help="Embeddings .npy file for the evaluation dataset.")
    parser.add_argument("--ids_csv",        required=True,
                        help="CSV with 'ID' column matching embedding order.")
    parser.add_argument("--model_path",     required=True,
                        help="Path to the Keras model file.")
    parser.add_argument("--output_csv",     required=True,
                        help="Output CSV with predictions.")
    parser.add_argument("--batch_size",     type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    X_eval = np.load(args.embeddings_npy)
    X_eval = X_eval.reshape(X_eval.shape[0], 255, 1024, 1)

    ids = pd.read_csv(args.ids_csv)["ID"].tolist()

    model = keras.models.load_model(args.model_path)
    preds = model.predict(X_eval, batch_size=args.batch_size)

    pred_labels = convert_preds(preds)
    pred_probs  = preds.flatten().tolist()

    results = pd.DataFrame({
        "ID":                ids,
        "LMPred_pred_prob":  pred_probs,
        "LMPred_pred_label": [int(x) for x in pred_labels],
    })
    results.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
