#!/usr/bin/env nextflow

include { plapdTraining } from './modules/plapd_training.nf'

// Pipeline parameters
params {
    tools_dir: Path
}

workflow {

    main:
    /*
     * TRAINING INPUT CHANNELS
    */    
    plapd_train_csv_ch = channel.fromPath("${params.tools_dir}/PLAPD/training_data.csv")
    plapd_val_csv_ch = channel.fromPath("${params.tools_dir}/PLAPD/val_data.csv")
    /*
     * TRAINING
    */
    plapdTraining(plapd_train_csv_ch, plapd_val_csv_ch)

    publish:
    plapd_model = plapdTraining.out.model
}

output {
    plapd_model {
        path { 'models/plapd' }
    }
}
