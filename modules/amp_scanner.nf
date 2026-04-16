/*
 * AMP Scanner
 * GitHub: https://github.com/dan-veltri/amp-scanner-v2
 * DockerHub: dveltri/ascan2:orig
 * Note: it requires a filtered FASTA with lengths >= 10 aa
*/

process ampScanner {

    container 'dveltri/ascan2:orig'

    input:
    path input_fasta

    output:
    path 'amp_scanner_predictions.csv', emit: predictions
    path 'amp_scanner_candidates.fasta', emit: candidates
    
    script:
    """
    python /app/amp_scanner_v2_predict_tf1.py \
        -fasta ${input_fasta} \
        -model /app/trained-models/OriginalPaper_081917_FULL_MODEL.h5 \
        -preds amp_scanner_predictions.csv \
        -candidates amp_scanner_candidates.fasta
    """
}