/*
 * PyAMPA, AMPValidate model
 * GitHub: https://github.com/SysBioUAB/PyAMPA
 */

process pyampa {
    
    container 'alvaromaximo/pyampa:1.0'

    input:
    path input_fasta

    output:
    path 'ampvalidate_predictions.csv', emit: predictions

    script:
    """
    python /app/pyampa/ampvalidate_predict.py \
        --fasta '${input_fasta}'
    """
}
