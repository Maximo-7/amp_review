/*
 * Macrel
 * GitHub: https://github.com/BigDataBiology/macrel
 * Note: --keep-negatives is required to get predictions for all sequences,
 *     not only those predicted as AMPs (prob > 0.5)
 */

process macrel {

    container 'alvaromaximo/macrel:1.0'

    input:
    path input_fasta

    output:
    path 'macrel_out/README.md', emit: readme
    path 'macrel_predictions.tsv', emit: predictions

    script:
    """
    macrel peptides \
        --fasta ${input_fasta} \
        --output macrel_out \
        --keep-negatives

    gunzip -c macrel_out/macrel.out.prediction.gz > macrel_predictions.tsv
    """  
}
