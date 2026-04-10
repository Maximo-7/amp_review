# Adapted from 1_Create_Word_Embeddings.ipynb
# Source: https://github.com/williamdee1/LMPred_AMP_Prediction
# Changes: removed Google Colab dependencies, added argparse CLI interface.
# The model with the best reported performance (T5 trained on UniRef50) was selected.

import argparse
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from tqdm.auto import tqdm
import pandas as pd
import numpy as np


# ─── Language model embedding class (original) ────────────────────────────────

class LM_EMBED:

    def __init__(self, language_model, max_len, rare_aa):
        self.lang_model = language_model
        self.max_len = max_len
        self.rare_aa = rare_aa

        if self.lang_model == 'T5-XL-UNI':
            self.tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(
                "Rostlab/prot_t5_xl_uniref50")
            gc.collect()

    def extract_word_embs(self, seq_df, filename, batch_size=10):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model = self.model.eval()

        seqs_list  = seq_df.Sequence.to_list()
        seqs_spaced = self.add_spaces(seqs_list)

        if self.rare_aa:
            seqs_spaced = [re.sub(r"[UZOB]", "X", sequence)
                        for sequence in seqs_spaced]

        n_seqs = len(seqs_spaced)
        dim1   = self.max_len - 2  # 255
        dim2   = 1024              # T5 hidden size

        # Pre-allocate output array on disk
        out = np.lib.format.open_memmap(
            filename, mode='w+', dtype='float32', shape=(n_seqs, dim1, dim2)
        )

        idx = 0
        for batch_start in range(0, n_seqs, batch_size):
            batch = seqs_spaced[batch_start:batch_start + batch_size]
            if batch_start % 1000 == 0:
                print(f"Processing sequences {batch_start}-{batch_start + len(batch)}...")

            ids = self.tokenizer.batch_encode_plus(
                batch,
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_len
            )

            input_ids      = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad():
                embeddings = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )[0]
                emb_array = embeddings.cpu().numpy()

            for seq_num in range(len(emb_array)):
                seq_len = int((attention_mask[seq_num] == 1).sum())
                seq_emd = emb_array[seq_num][:seq_len - 1]
                padded  = np.zeros((dim1, dim2), dtype='float32')
                padded[:seq_emd.shape[0], :seq_emd.shape[1]] = seq_emd
                out[idx] = padded
                idx += 1

            del input_ids, attention_mask, embeddings, emb_array
            torch.cuda.empty_cache()

        out.flush()
        print("Saving Embeddings...")

    def add_spaces(self, df_col):
        return [" ".join(x) for x in df_col]

    def extract_features(self, emb_res, att_msk):
        features = []

        for seq_num in range(len(emb_res)):
            seq_len = (att_msk[seq_num] == 1).sum()

            if self.lang_model in ['T5-XL-BFD', 'T5-XL-UNI']:
                seq_emd = emb_res[seq_num][:seq_len - 1]

            features.append(seq_emd)

        features_arr = np.array(features, dtype=object)
        return features_arr

    def pad(self, features):
        dim1 = self.max_len - 2
        dim2 = features[0].shape[1]

        padded_list = []

        for i in range(len(features)):
            if i % 100 == 0:
                print("Padding Batch: ", i)

            all_zeros = np.zeros((dim1, dim2))
            all_zeros[:features[i].shape[0], :features[i].shape[1]] = features[i]

            padded_list.append(all_zeros)
            
        padded_arr = np.stack(padded_list, axis=0)
        return padded_arr


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create T5XL-UNI embeddings from a CSV of peptide sequences."
    )
    parser.add_argument(
        "--input_csv", required=True,
        help="CSV file with a 'Sequence' column."
    )
    parser.add_argument(
        "--output_npy", required=True,
        help="Output .npy file path for the embedding array."
    )
    parser.add_argument(
        "--max_len", type=int, default=257,
        help="Max sequence length + 2 special tokens (default: 257)."
    )
    parser.add_argument("--batch_size", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    seq_df = pd.read_csv(args.input_csv)
    embedder = LM_EMBED('T5-XL-UNI', args.max_len, rare_aa=True)
    embedder.extract_word_embs(seq_df, args.output_npy, args.batch_size)


if __name__ == "__main__":
    main()
