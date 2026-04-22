#!/usr/bin/env nextflow

include { ktAmppredTraining } from './modules/kt_amppred_training.nf'

// Pipeline parameters
params {
    tools_dir: Path
}

workflow {

    main:
    /*
     * TRAINING INPUT CHANNELS
    */
    kt_amppred_train_tsv_ch = channel.fromPath("${params.tools_dir}/KT-AMPpred/amp_train.tsv")
    kt_amppred_test_tsv_ch = channel.fromPath("${params.tools_dir}/KT-AMPpred/amp_test.tsv")
    /*
     * TRAINING
    */
    ktAmppredTraining(kt_amppred_train_tsv_ch, kt_amppred_test_tsv_ch)

    publish:
    kt_amppred_model_dir = ktAmppredTraining.out.model_dir
}

output {
    kt_amppred_model_dir {
        path { 'models/kt_amppred' }
    }
}
