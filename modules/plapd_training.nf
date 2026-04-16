/*
 * PLAPD
 * GitHub: https://github.com/lichaozhang2/PLAPD
 * Train PLAPD classifier, leveraging a pre-trained ESM2 model.
 * Optional step: skipped if params.plapd_skip_training = true
 */

 process plapdTraining {

    label 'process_gpu'

    container 'alvaromaximo/plapd:1.0'

    input:
    path train_csv
    path val_csv

    output:
    path 'my_best_model_without_embedding.pth', emit: model

    script:
    """
    python /app/plapd/MyTrain.py \
        --train_csv '${train_csv}' \
        --val_csv '${val_csv}'
    """
}
