#!/usr/bin/env nextflow

// Include modules
include { lmpredEmbeddings as lmpredEmbTrain} from './modules/lmpred_embeddings.nf'
include { lmpredEmbeddings as lmpredEmbVal} from './modules/lmpred_embeddings.nf'
include { lmpredTraining } from './modules/lmpred_training.nf'

// Pipeline parameters
params {
    tools_dir: Path
}

workflow {

    main:
    /*
     * TRAINING INPUT CHANNELS
    */
    lmpred_x_train_csv_ch = channel.fromPath("${params.tools_dir}/LMPred/X_train.csv")
    lmpred_x_val_csv_ch   = channel.fromPath("${params.tools_dir}/LMPred/X_val.csv")
    lmpred_y_train_csv_ch = channel.fromPath("${params.tools_dir}/LMPred/y_train.csv")
    lmpred_y_val_csv_ch   = channel.fromPath("${params.tools_dir}/LMPred/y_val.csv")
    /*
     * TRAINING
    */
    lmpredEmbTrain(lmpred_x_train_csv_ch)
    lmpredEmbVal(lmpred_x_val_csv_ch)
    lmpredTraining(lmpredEmbTrain.out.embeddings, lmpredEmbVal.out.embeddings, lmpred_y_train_csv_ch, lmpred_y_val_csv_ch)

    publish:
    lmpred_model = lmpredTraining.out.model
    lmpred_training_plots = lmpredTraining.out.plots
}

output {
    lmpred_model {
        path { 'models/lmpred' }
    }
    lmpred_training_plots {
        path { 'models/lmpred' }
    }
}
