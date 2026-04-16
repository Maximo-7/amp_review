/*
 * PepNet - Predict using ProtT5-XL-UniRef50 embeddings
 * GitHub: https://github.com/hjy23/PepNet
 */

process pepnet {

    label 'process_gpu'

    container 'alvaromaximo/pepnet:1.0'

    input:
    path embeddings_h5
    path input_fasta

    output:
    path 'AMP_prediction_result.csv', emit: predictions

    script:
    """
    python /app/pepnet/script/predict.py \
        -type AMP \
        -output_path ./ \
        -test_fasta '${input_fasta}' \
        -feature_file '${embeddings_h5}'
    """
}