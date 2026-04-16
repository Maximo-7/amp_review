/*
 * LMPred - Create T5XL-UNI embeddings
 * GitHub: https://github.com/williamdee1/LMPred_AMP_Prediction
 * Used for both training data and evaluation dataset embeddings.
 */

process lmpredEmbeddings {

    label 'process_gpu'

    container 'alvaromaximo/lmpred:1.0'

    input:
    path input_csv

    output:
    path "${input_csv.baseName}_embeddings.npy", emit: embeddings

    script:
    """
    python /app/lmpred/lmpred_create_embeddings.py \
        --input_csv  '${input_csv}' \
        --output_npy '${input_csv.baseName}_embeddings.npy'
    """
}
