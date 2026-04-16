/*
 * MultiAMP, predict from FASTA file (no PDB)
 * GitHub: https://github.com/jiayili11/multi-amp
 */

process multiamp {

    label 'process_gpu'

    container 'alvaromaximo/multiamp:1.0'

    input:
    path input_fasta

    output:
    path 'multiamp_predictions.csv', emit: predictions

    script:
    """
    python predict.py --gpu 0 \
        --model_path /app/multiamp/checkpoints/best_model_overall.pth \
        --fasta_path '${input_fasta}' \
        --output_path multiamp_predictions.csv
    """
}