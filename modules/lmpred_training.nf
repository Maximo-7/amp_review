/*
 * LMPred - Train two-layer CNN classifier
 * GitHub: https://github.com/williamdee1/LMPred_AMP_Prediction
 * Optional step: skipped if params.lmpred_skip_training = true
 */

process lmpredTraining {

    label 'process_gpu'

    container 'alvaromaximo/lmpred:1.0'

    input:
    path x_train_npy
    path x_val_npy
    path y_train_csv
    path y_val_csv

    output:
    path 'lmpred_best_model.keras', emit: model
    path 'lmpred_training_curves.png', emit: plots

    script:
    """
    python /app/lmpred/lmpred_train.py \
        --x_train_npy '${x_train_npy}' \
        --x_val_npy   '${x_val_npy}' \
        --y_train_csv '${y_train_csv}' \
        --y_val_csv   '${y_val_csv}' \
        --model_path  lmpred_best_model.keras \
        --plots_path  lmpred_training_curves.png
    """
}
