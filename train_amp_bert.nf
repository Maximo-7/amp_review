#!/usr/bin/env nextflow

include { ampBertTraining } from './modules/amp_bert_training.nf'

// Pipeline parameters
params {
    tools_dir: Path
}

workflow {

    main:
    /*
     * TRAINING INPUT CHANNELS
    */
    amp_bert_train_csv_ch = channel.fromPath("${params.tools_dir}/AMP-BERT/all_veltri.csv")
    /*
     * TRAINING
    */
    ampBertTraining(amp_bert_train_csv_ch)

    publish:    
    amp_bert_model_dir = ampBertTraining.out.model_dir
}

output {
    amp_bert_model_dir {
        path { 'models/amp_bert' }
    }
}
