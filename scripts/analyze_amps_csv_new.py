import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Dictionary of free interaction energy values (kcal/mol) used to calculate the Boman index
boman_values = {
    'A': 0.17, 'C': -0.24, 'D': -0.77, 'E': -0.64, 'F': -0.22,
    'G': 0.01, 'H': -0.96, 'I': -0.31, 'K': -0.99, 'L': -0.21,
    'M': -0.23, 'N': -0.60, 'P': 0.45, 'Q': -0.69, 'R': -2.56,
    'S': -0.02, 'T': -0.04, 'V': -0.07, 'W': -0.88, 'Y': -0.33
}

def calculate_boman_index(sequence: str) -> float:
    """
    Calculate the Boman index for an amino acid sequence.

    The Boman index estimates the potential of a peptide to bind
    to proteins, based on free energy of interaction values for each residue.
    """
    return sum(boman_values.get(aa, 0) for aa in sequence) / len(sequence) if sequence else 0.0

def analyze_sequence(seq_id: str, sequence: str):
    """
    Analyze physicochemical properties of an amino acid sequence,
    preserving its original FASTA ID.

    If the sequence is invalid or empty, it will still be included in the
    output but with NaN values for all properties.
    """
    # Clean sequence: keep only letters, remove gaps/spaces, convert to uppercase
    sequence = "".join([aa for aa in str(sequence) if aa.isalpha()]).strip().upper()
    
    if not sequence:
        print(f"⚠️ Warning: Empty sequence in ID: {seq_id}. It will be kept with NaN values.")
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
    
    if any(aa not in boman_values for aa in sequence):
        print(f"⚠️ Warning: Invalid characters found in sequence ID: {seq_id}. It will be kept with NaN values.")
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
        return {
            "ID": seq_id,
            "Sequence": sequence,
            "Molecular Weight": analyzer.molecular_weight(),
            "Net Charge (pH 7)": analyzer.charge_at_pH(7.0),
            "Aromaticity": analyzer.aromaticity(),
            "Instability Index": analyzer.instability_index(),
            "Isoelectric Point": analyzer.isoelectric_point(),
            "GRAVY": analyzer.gravy(),
            "Boman Index": calculate_boman_index(sequence)
        }
    except Exception as e:
        print(f"❌ Error analyzing sequence with ID {seq_id}: {e}. It will be kept with NaN values.")
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
        # Read FASTA file using Biopython, ignoring non-ASCII characters
        with open(input_path, "r", encoding="ascii", errors="ignore") as handle:
            records = list(SeqIO.parse(handle, "fasta"))
    except Exception as e:
        print(f"❌ Error reading FASTA file: {e}")
        return

    # Analyze each sequence (all IDs will be kept, valid or invalid)
    results = [analyze_sequence(record.id, record.seq) for record in records]

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
    parser = argparse.ArgumentParser(description="Analyze peptide sequences from a FASTA file, preserving sequence IDs.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input FASTA file.")
    parser.add_argument("-o", "--output", default="AMPs_properties_from_fasta.csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    process_fasta(args.input, args.output)

if __name__ == "__main__":
    main()
