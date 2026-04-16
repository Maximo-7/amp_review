/*
 * AMPlify
 * GitHub: https://github.com/bcgsc/AMPlify
 */

 process amplify {

    container 'alvaromaximo/amplify:1.0'

    input:
    path input_fasta

    output:
    path 'AMPlify_*_results_*.tsv', emit: predictions

    script:
    """
    AMPlify \
    -m balanced \
    -s '${input_fasta}' \
    -of 'tsv'
    """
 }