/*
 * Ma et al. (2022) - BERT model prediction
 * Input: FASTA file (directly, no formatting needed)
 */

process maEtAlBert {

    container 'alvaromaximo/ma_et_al_bert:1.0'

    input:
    path input_fasta

    output:
    path 'bert_predictions.txt', emit: predictions

    script:
    """
    python /app/ma_et_al/script/prediction_bert.py \
        '${input_fasta}' \
        bert_predictions.txt
    """
}
