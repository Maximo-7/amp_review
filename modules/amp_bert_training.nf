/*
 * AMP-BERT
 * GitHub: https://github.com/GIST-CSBL/AMP-BERT
 * Fine-tuned BertForSequenceClassification model (prot_bert_bfd backbone).
 * Optional step: skipped if params.amp_bert_skip_training = true
 */

process ampBertTraining {

    label 'process_gpu'

    container 'alvaromaximo/amp_bert:1.1'

    input:
    path input_csv

    output:
    path 'amp_bert_model', emit: model_dir

    script:
    """
    python /app/amp_bert/amp_bert_train.py \
        '${input_csv}'
    """
}
