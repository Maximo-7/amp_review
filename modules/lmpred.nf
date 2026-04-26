/*
 * LMPred - Predict on evaluation dataset
 * GitHub: https://github.com/williamdee1/LMPred_AMP_Prediction
 */

process lmpred {

    label 'gpu0'

    container 'alvaromaximo/lmpred:1.2-model'

    input:
    path embeddings_npy
    path ids_csv

    output:
    path 'lmpred_predictions.csv', emit: predictions

    script:
    """
    python /app/lmpred/lmpred_predict.py \
        --embeddings_npy '${embeddings_npy}' \
        --ids_csv        '${ids_csv}' \
        --model_path     /app/lmpred/model/T5XL_UNI_best_model.epoch06-loss0.28.keras \
        --output_csv     lmpred_predictions.csv
    """
}
