/*
 * amPEPpy
 * GitHub: https://github.com/tlawrence3/amPEPpy
 */

process ampeppy {

    container 'alvaromaximo/ampeppy:1.1'

    input:
    path input_fasta

    output:
    path 'ampeppy_predictions.tsv', emit: predictions
    path '*_scored_features.csv'  , emit: features

    script:
    """
    ampep predict \
        -m /app/ampeppy/pretrained_models/amPEP.model \
        -i "${input_fasta}" \
        -o ampeppy_predictions.tsv \
        --seed 2012
    """   
}