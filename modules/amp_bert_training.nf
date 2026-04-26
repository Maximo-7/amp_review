/*
 * AMP-BERT
 * GitHub: https://github.com/GIST-CSBL/AMP-BERT
 * Fine-tuned BertForSequenceClassification model (prot_bert_bfd backbone).
 * Module for train.nf
 */

process ampBertTraining {

    label 'gpu0'

    container 'alvaromaximo/amp_bert:1.2-base'

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
