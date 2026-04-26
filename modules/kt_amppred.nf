/*
 * KT-AMPpred - Predict using the fine-tuned AMP classifier
 * GitHub: https://github.com/liangxiaodata/AMPpred
 */

process ktAmppred {

    label 'gpu1'

    container 'alvaromaximo/kt_amppred:1.1-model'

    input:
    path input_fasta

    output:
    path 'kt_amppred_predictions.csv', emit: predictions

    script:
    """
    python /app/kt_amppred/kt_amppred_predict.py \
        --fasta '${input_fasta}' \
        --model_dir /app/kt_amppred/model \
        --classifier /app/kt_amppred/model/classifier_weights.pth
    """
}
