/*
 * PLAPD - Predict using the trained AMP classifier
 * GitHub: https://github.com/lichaozhang2/PLAPD
 */

process plapd {

    label 'gpu1'

    container 'alvaromaximo/plapd:1.1-model'

    input:
    path input_fasta

    output:
    path 'plapd_predictions.csv', emit: predictions

    script:
    """
    python /app/plapd/MyModel.py \
        --input_fasta '${input_fasta}' \
        --model /app/plapd/model/my_best_model_without_embedding.pth
    """
}
