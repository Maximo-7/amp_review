/*
 * Ma et al. (2022) - Combine predictions from all three models
 * Consensus: AMP only if all three models predict AMP (prob > 0.5)
 */

process maEtAlCombine {

    container 'alvaromaximo/ma_et_al_att_lstm:1.0'

    input:
    path att_predictions
    path lstm_predictions
    path bert_predictions
    path input_fasta

    output:
    path 'ma_et_al_result.txt', emit: result

    script:
    """
    perl /app/ma_et_al/script/result_maximo.pl \
        '${att_predictions}' \
        '${lstm_predictions}' \
        '${bert_predictions}' \
        '${input_fasta}' > ma_et_al_result.txt
    """
}
