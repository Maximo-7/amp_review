/*
 * Ma et al. (2022) - Format FASTA for attention and LSTM models
 * GitHub: https://github.com/mayuefine/c_AMPs-prediction
 * Note: sequences with ambiguous amino acids (B, J, O, U, X, Z) are skipped
 */

process maEtAlFormat {

    container 'alvaromaximo/ma_et_al_att_lstm:1.0'

    input:
    path input_fasta

    output:
    path 'sequences_formatted.txt', emit: formatted

    script:
    """
    perl /app/ma_et_al/script/format.pl '${input_fasta}' none > sequences_formatted.txt
    """
}
