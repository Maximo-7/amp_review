/*
 * AMP-BERT - Predict on evaluation dataset
 * GitHub: https://github.com/williamdee1/LMPred_AMP_Prediction
 * Fine-tuned BertForSequenceClassification model (prot_bert_bfd backbone).
 */

process ampBert {

    label 'process_gpu'

    container 'alvaromaximo/amp_bert:1.3'

    input:
    path input_csv

    output:
    path 'amp_bert_predictions.csv', emit: predictions

    script:
    """
    python /app/amp_bert/amp_bert_predict.py \
        --input_csv  '${input_csv}' \
        --model_dir  /app/amp_bert/model \
        --output_csv amp_bert_predictions.csv
    """
}
