/*
 * PepNet - Create ProtT5-XL-UniRef50, half precision model embeddings
 * HuggingFace: https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main
 * GitHub: https://github.com/agemagician/ProtTrans (for prott5_embedder.py)
 */

process pepnetEmbeddings {

    label 'process_gpu'

    container 'alvaromaximo/pepnet:1.0'

    input:
    path input_fasta

    output:
    path "${input_fasta.baseName}_embeddings.h5", emit: embeddings

    script:
    """
    python /app/prottrans/prott5_embedder.py \
        --input  '${input_fasta}' \
        --output '${input_fasta.baseName}_embeddings.h5' \
        --model /app/prottrans/prot_t5_xl_half_uniref50-enc
    """
}