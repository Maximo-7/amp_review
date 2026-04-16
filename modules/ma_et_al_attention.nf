/*
 * Ma et al. (2022) - Attention model prediction
 * Input: sequences_formatted.txt from maEtAlFormat
 */

process maEtAlAttention {

    container 'alvaromaximo/ma_et_al_att_lstm:1.0'

    input:
    path formatted_txt

    output:
    path 'att_predictions.txt', emit: predictions

    script:
    """
    python /app/ma_et_al/script/prediction_attention.py \\
        '${formatted_txt}' \\
        att_predictions.txt
    """
}
