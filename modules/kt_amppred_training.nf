/*
 * KT-AMPpred, ProtT5-based AMP classifier
 * GitHub: https://github.com/liangxiaodata/AMPpred
 * Module for train.nf
 * Note: test_tsv is not used to select the best model
 */

process ktAmppredTraining {

    label 'gpu0'

    container 'alvaromaximo/kt_amppred:1.1-base'

    input:
    path train_tsv
    path test_tsv

    output:
    path 'finetune_peptide_model', emit: model_dir

    script:
    """
    python /app/kt_amppred/main_amp.py \
        --train_tsv '${train_tsv}' \
        --test_tsv '${test_tsv}'
    """
}
