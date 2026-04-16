/*
 * DLFea4AMPGen, ABP-MPB model using CPU
 * GitHub: https://github.com/hgao12345/DLFea4AMPGen
 */

 process dlfea4ampgen {

    container 'alvaromaximo/dlfea4ampgen:1.0'

    input:
    path input_csv

    output:
    path 'dlfea4ampgen_predictions.csv', emit: predictions
    path 'log.log', emit: stdout
    path 'sys.log', emit: stderr

    script:
    """
    mkdir -p /app/dlfea4ampgen/output_dir

    python /app/dlfea4ampgen/Finetune/mpbert_classification.py \
        --config_path /app/dlfea4ampgen/Finetune/config_1024.yaml \
        --load_checkpoint_url /app/dlfea4ampgen/models/ABP_Best_Model.ckpt \
        --do_predict True \
        --description classification \
        --num_class 2 \
        --device_id 0 \
        --device_target CPU \
        --vocab_file /app/dlfea4ampgen/Finetune/vocab_v2.txt \
        --data_url '${input_csv}' \
        --output_url /app/dlfea4ampgen/output_dir \
        --return_sequence False \
        --return_csv True 1> log.log 2> sys.log
    mv '/app/dlfea4ampgen/output_dir/${input_csv.baseName}_predict_result.csv' dlfea4ampgen_predictions.csv
    """
 }