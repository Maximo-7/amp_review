import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import peptides


def analyze_sequence(seq_id: str, sequence: str):
    """
    Analyze physicochemical properties of an amino acid sequence,
    preserving its original FASTA ID.

    If the sequence is invalid or empty, it will still be included in the
    output but with NaN values for all properties.
    """

    # Clean sequence: keep only letters, remove gaps/spaces, convert to uppercase
    sequence = "".join(
        aa for aa in str(sequence) if aa.isalpha()
    ).strip().upper()

    if not sequence:
        print(
            f"⚠️ Warning: Empty sequence in ID: {seq_id}. "
            "It will be kept with NaN values."
        )

        return {
            "ID": seq_id,
            "Sequence": None,
            "Molecular Weight": np.nan,
            "Net Charge (pH 7)": np.nan,
            "Aromaticity": np.nan,
            "Instability Index": np.nan,
            "Isoelectric Point": np.nan,
            "GRAVY": np.nan,
            "Boman Index": np.nan
        }

    # Standard amino acids accepted by ProteinAnalysis / peptides
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    if any(aa not in valid_amino_acids for aa in sequence):
        print(
            f"⚠️ Warning: Invalid characters found in sequence ID: {seq_id}. "
            "It will be kept with NaN values."
        )

        return {
            "ID": seq_id,
            "Sequence": sequence,
            "Molecular Weight": np.nan,
            "Net Charge (pH 7)": np.nan,
            "Aromaticity": np.nan,
            "Instability Index": np.nan,
            "Isoelectric Point": np.nan,
            "GRAVY": np.nan,
            "Boman Index": np.nan
        }

    try:
        analyzer = ProteinAnalysis(sequence)

        # peptides.py object
        peptide = peptides.Peptide(sequence)

        return {
            "ID": seq_id,
            "Sequence": sequence,
            "Molecular Weight": analyzer.molecular_weight(),
            "Net Charge (pH 7)": analyzer.charge_at_pH(7.0),
            "Aromaticity": analyzer.aromaticity(),
            "Instability Index": analyzer.instability_index(),
            "Isoelectric Point": analyzer.isoelectric_point(),
            "GRAVY": analyzer.gravy(),

            # Boman index calculated using peptides.py
            "Boman Index": peptide.boman()
        }

    except Exception as e:
        print(
            f"❌ Error analyzing sequence with ID {seq_id}: {e}. "
            "It will be kept with NaN values."
        )

        return {
            "ID": seq_id,
            "Sequence": sequence,
            "Molecular Weight": np.nan,
            "Net Charge (pH 7)": np.nan,
            "Aromaticity": np.nan,
            "Instability Index": np.nan,
            "Isoelectric Point": np.nan,
            "GRAVY": np.nan,
            "Boman Index": np.nan
        }


def process_fasta(input_path: str, output_path: str):
    """
    Process a FASTA file and calculate sequence properties,
    saving results into a CSV file.
    """

    try:
        with open(
            input_path,
            "r",
            encoding="ascii",
            errors="ignore"
        ) as handle:

            records = list(SeqIO.parse(handle, "fasta"))

    except Exception as e:
        print(f"❌ Error reading FASTA file: {e}")
        return

    results = [
        analyze_sequence(record.id, record.seq)
        for record in records
    ]

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

        print(f"✅ CSV file saved at: {output_path}")

    else:
        print("⚠️ No sequences found for processing.")


def main():
    """
    Command-line interface for the script.

    Usage:
        python script.py -i input.fasta -o output.csv
    """

    parser = argparse.ArgumentParser(
        description=(
            "Analyze peptide sequences from a FASTA file, "
            "preserving sequence IDs."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input FASTA file."
    )

    parser.add_argument(
        "-o",
        "--output",
        default="AMPs_properties_from_fasta.csv",
        help="Path to the output CSV file."
    )

    args = parser.parse_args()

    process_fasta(args.input, args.output)


if __name__ == "__main__":
    main()
