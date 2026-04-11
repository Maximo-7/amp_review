# run_cli_maximo.py
# Command-line interface for AMPFinder, bypassing the GUI

import os
import argparse

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="AMPFinder CLI - Predict antimicrobial peptides without GUI")
    parser.add_argument("--input", required=True, help="Path to input FASTA file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for AMP classification (default: 0.5)")
    parser.add_argument("--mode", choices=["identify", "function", "both"], default="identify",
                        help="Prediction mode: 'identify' (AMP or not), 'function' (AMP activity), 'both' (default: identify)")
    parser.add_argument("--output", default=None, help="Path to output file (default: print to stdout)")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        return

    # Read input FASTA file
    info = open(args.input).read()

    # Import AMPFinder functions without launching the Qt GUI
    # We mock QApplication and QUiLoader so PySide2 never tries to open a display
    import unittest.mock as mock
    with mock.patch("PySide2.QtWidgets.QApplication"), \
         mock.patch("PySide2.QtUiTools.QUiLoader"):
        import main_windows as mw

    # Collect results
    results = []

    if args.mode in ("identify", "both"):
        print(f"[*] Running AMP identification with threshold={args.threshold}...")
        ress = mw.predict_amp(info, str(args.threshold))
        results.append("=== AMP Identification Results ===")
        results.append("ID,Sequence,Probability,Label_num,Label")
        results.extend(ress)

    if args.mode in ("function", "both"):
        print(f"[*] Running AMP function prediction with threshold={args.threshold}...")
        ress = mw.predict_fun(info, str(args.threshold))
        results.append("=== AMP Function Prediction Results ===")
        results.append("ID,Sequence,bacterial,viral,parasital,HIV,cancer,MRSA,fungal,endotoxin,biofilm,Chemotactic")
        results.extend(ress)

    # Write results to file or print to stdout
    results = [r.strip() for r in results]
    output_text = "\n".join(results)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"[*] Results saved to '{args.output}'")
    else:
        print(output_text)

if __name__ == "__main__":
    main()