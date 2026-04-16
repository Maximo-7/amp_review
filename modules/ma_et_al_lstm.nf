/*
 * Ma et al. (2022) - LSTM model prediction
 * Input: sequences_formatted.txt from maEtAlFormat
 */

process maEtAlLstm {

    container 'alvaromaximo/ma_et_al_att_lstm:1.1'

    input:
    path formatted_txt

    output:
    path 'lstm_predictions.txt', emit: predictions

    script:
    """
    python /app/ma_et_al/script/prediction_lstm.py \
        '${formatted_txt}' \
        lstm_predictions.txt
    """
}
