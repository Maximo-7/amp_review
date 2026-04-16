/*
 * KT-AMPpred - Predict using the fine-tuned AMP classifier
 * GitHub: https://github.com/liangxiaodata/AMPpred
 */

process ktAmppred {

    label 'process_gpu'

    container 'alvaromaximo/kt_amppred:1.0'

    input:
    path input_fasta
    path model_dir

    output:
    path 'kt_amppred_predictions.csv', emit: predictions

    script:
    """
    python /app/kt_amppred/kt_amppred_predict.py \
        --fasta '${input_fasta}' \
        --model_dir '${model_dir}' \
        --classifier '${model_dir}/classifier_weights.pth'
    """
}
