/*
 * AMPFinder
 * GitHub: https://github.com/abcair/AMPFinder
 * Note: run_cli_maximo.py bypasses the GUI
 */

process ampfinder {

    container 'alvaromaximo/ampfinder:1.2'

    input:
    path input_fasta

    output:
    path 'ampfinder_predictions.csv', emit: predictions

    script:
    """
    python /app/ampfinder/run_cli_maximo.py \
        --input '${input_fasta}' \
        --output ampfinder_predictions.csv \
    """   
}