#!/usr/bin/env nextflow

// Include modules
include { ampScanner } from './modules/amp_scanner.nf'
include { macrel } from './modules/macrel.nf'
include { ampeppy } from './modules/ampeppy.nf'
include { lmpredEmbeddings as lmpredEmbTrain} from './modules/lmpred_embeddings.nf'
include { lmpredEmbeddings as lmpredEmbVal} from './modules/lmpred_embeddings.nf'
include { lmpredEmbeddings as lmpredEmbTest} from './modules/lmpred_embeddings.nf'
include { lmpredTraining } from './modules/lmpred_training.nf'
include { lmpred } from './modules/lmpred.nf'
include { amplify } from './modules/amplify.nf'
include { maEtAlFormat } from './modules/ma_et_al_format.nf'
include { maEtAlAttention } from './modules/ma_et_al_attention.nf'
include { maEtAlLstm } from './modules/ma_et_al_lstm.nf'
include { maEtAlBert } from './modules/ma_et_al_bert.nf'
include { maEtAlCombine } from './modules/ma_et_al_combine.nf'
include { ampBertTraining } from './modules/amp_bert_training.nf'
include { ampBert } from './modules/amp_bert.nf'
include { ampfinder } from './modules/ampfinder.nf'
include { pyampa } from './modules/pyampa.nf'
include { pepnetEmbeddings } from './modules/pepnet_embeddings.nf'
include { pepnet } from './modules/pepnet.nf'
include { ktAmppredTraining } from './modules/kt_amppred_training.nf'
include { ktAmppred } from './modules/kt_amppred.nf'
include { plapdTraining } from './modules/plapd_training.nf'
include { plapd } from './modules/plapd.nf'
include { dlfea4ampgen } from './modules/dlfea4ampgen.nf'
include { multiamp } from './modules/multiamp.nf'

// Pipeline parameters
params {
    eval_dir: Path
    tools_dir: Path
    models_dir: Path
    lmpred_skip_training: Boolean
    amp_bert_skip_training: Boolean
    kt_amppred_skip_training: Boolean
    plapd_skip_training: Boolean
}

workflow {

    main:
    /*
     * EVALUATION INPUT CHANNELS
    */
    ev_fasta_ch = channel.fromPath("${params.eval_dir}/evaluation_dataset.fasta")
    ev_fasta_geq_10aa_ch = channel.fromPath("${params.eval_dir}/evaluation_dataset_geq_10aa.fasta")
    x_test_csv_ch = channel.fromPath("${params.eval_dir}/x_test_maximo.csv")
    x_test_wo_length_csv_ch = channel.fromPath("${params.eval_dir}/x_test_maximo_wo_length.csv")
    /*
     * OPTIONAL TRAININGS
     */
    // LMPred
    if (!params.lmpred_skip_training) {
        lmpred_x_train_csv_ch = channel.fromPath("${params.tools_dir}/LMPred/X_train.csv")
        lmpred_x_val_csv_ch   = channel.fromPath("${params.tools_dir}/LMPred/X_val.csv")
        lmpred_y_train_csv_ch = channel.fromPath("${params.tools_dir}/LMPred/y_train.csv")
        lmpred_y_val_csv_ch   = channel.fromPath("${params.tools_dir}/LMPred/y_val.csv")       

        lmpredEmbTrain(lmpred_x_train_csv_ch)
        lmpredEmbVal(lmpred_x_val_csv_ch)
        lmpredTraining(lmpredEmbTrain.out.embeddings, lmpredEmbVal.out.embeddings, lmpred_y_train_csv_ch, lmpred_y_val_csv_ch)
        lmpred_model_ch = lmpredTraining.out.model
    } else {
        lmpred_model_ch = channel.fromPath("${params.models_dir}/lmpred/T5XL_UNI_best_model.epoch06-loss0.28.keras")
    }
    // AMP-BERT
    if (!params.amp_bert_skip_training) {
        amp_bert_train_csv_ch = channel.fromPath("${params.tools_dir}/AMP-BERT/all_veltri.csv")

        ampBertTraining(amp_bert_train_csv_ch)
        amp_bert_model_ch = ampBertTraining.out.model_dir
    } else {
        amp_bert_model_ch = channel.fromPath("${params.models_dir}/amp_bert")
    }
    // KT-AMPpred
    if (!params.kt_amppred_skip_training) {
        kt_amppred_train_tsv_ch = channel.fromPath("${params.tools_dir}/KT-AMPpred/amp_train.tsv")
        kt_amppred_test_tsv_ch = channel.fromPath("${params.tools_dir}/KT-AMPpred/amp_test.tsv")

        ktAmppredTraining(kt_amppred_train_tsv_ch, kt_amppred_test_tsv_ch)
        kt_amppred_model_ch = ktAmppredTraining.out.model_dir
    } else {
        kt_amppred_model_ch = channel.fromPath("${params.models_dir}/kt_amppred")
    }
    // PLAPD
    if (!params.plapd_skip_training) {
        plapd_train_csv_ch = channel.fromPath("${params.tools_dir}/PLAPD/training_data.csv")
        plapd_val_csv_ch = channel.fromPath("${params.tools_dir}/PLAPD/val_data.csv")

        plapdTraining(plapd_train_csv_ch, plapd_val_csv_ch)
        plapd_model_ch = plapdTraining.out.model
    } else {
        plapd_model_ch = channel.fromPath("${params.models_dir}/plapd/my_best_model_without_embedding.pth")
    }
    /*
     * PREDICTIONS
    */
    ampScanner(ev_fasta_geq_10aa_ch)
    macrel(ev_fasta_ch)
    ampeppy(ev_fasta_ch)
    lmpredEmbTest(x_test_csv_ch)
    lmpred(lmpredEmbTest.out.embeddings, x_test_csv_ch, lmpred_model_ch)
    amplify(ev_fasta_ch)
    maEtAlFormat(ev_fasta_ch)
    maEtAlAttention(maEtAlFormat.out.formatted)
    maEtAlLstm(maEtAlFormat.out.formatted)
    maEtAlBert(ev_fasta_ch)
    maEtAlCombine(maEtAlAttention.out.predictions,
        maEtAlLstm.out.predictions,
        maEtAlBert.out.predictions,
        ev_fasta_ch)
    ampBert(x_test_csv_ch, amp_bert_model_ch)
    ampfinder(ev_fasta_ch)
    pyampa(ev_fasta_ch)
    pepnetEmbeddings(ev_fasta_ch)
    pepnet(pepnetEmbeddings.out.embeddings, ev_fasta_ch)
    ktAmppred(ev_fasta_ch, kt_amppred_model_ch)
    plapd(ev_fasta_ch, plapd_model_ch)
    dlfea4ampgen(x_test_wo_length_csv_ch)
    multiamp(ev_fasta_ch)

    publish:
    amp_scanner_predictions = ampScanner.out.predictions
    macrel_predictions = macrel.out.predictions
    ampeppy_predictions = ampeppy.out.predictions
    lmpred_model = (!params.lmpred_skip_training) ? lmpredTraining.out.model : channel.empty()
    lmpred_training_plots = (!params.lmpred_skip_training) ? lmpredTraining.out.plots : channel.empty()
    lmpred_predictions = lmpred.out.predictions
    amplify_predictions = amplify.out.predictions
    ma_et_al_att_predictions = maEtAlAttention.out.predictions
    ma_et_al_lstm_predictions = maEtAlLstm.out.predictions
    ma_et_al_bert_predictions = maEtAlBert.out.predictions
    ma_et_al_result = maEtAlCombine.out.result
    amp_bert_model_dir = (!params.amp_bert_skip_training) ? ampBertTraining.out.model_dir : channel.empty()
    amp_bert_predictions = ampBert.out.predictions
    ampfinder_predictions = ampfinder.out.predictions
    pyampa_predictions = pyampa.out.predictions
    pepnet_predictions = pepnet.out.predictions
    kt_amppred_model_dir = (!params.kt_amppred_skip_training) ? ktAmppredTraining.out.model : channel.empty()
    kt_amppred_predictions = ktAmppred.out.predictions
    plapd_model = (!params.plapd_skip_training) ? plapdTraining.out.model : channel.empty()
    plapd_predictions = plapd.out.predictions
    dlfea4ampgen_predictions = dlfea4ampgen.out.predictions
    multiamp_predictions = multiamp.out.predictions
}

output {
    amp_scanner_predictions {
        path { 'tools_predictions/amp_scanner' }
    }
    macrel_predictions {
        path { 'tools_predictions/macrel' }
    }
    ampeppy_predictions {
        path { 'tools_predictions/ampeppy' }
    }
    lmpred_model {
        path { 'models/lmpred' }
    }
    lmpred_training_plots {
        path { 'models/lmpred' }
    }
    lmpred_predictions {
        path { 'tools_predictions/lmpred' }
    }
    amplify_predictions {
        path { 'tools_predictions/amplify' }
    }
    ma_et_al_att_predictions {
        path { 'tools_predictions/ma_et_al' }
    }
    ma_et_al_lstm_predictions {
        path { 'tools_predictions/ma_et_al' }
    }
    ma_et_al_bert_predictions {
        path { 'tools_predictions/ma_et_al' }
    }
    ma_et_al_result {
        path { 'tools_predictions/ma_et_al' }
    }
    amp_bert_model_dir {
        path { 'models/amp_bert' }
    }
    amp_bert_predictions {
        path { 'tools_predictions/amp_bert' }
    }
    ampfinder_predictions {
        path { 'tools_predictions/ampfinder' }
    }
    pyampa_predictions {
        path { 'tools_predictions/pyampa' }
    }
    pepnet_predictions {
        path { 'tools_predictions/pepnet' }
    }
    kt_amppred_model_dir {
        path { 'models/kt_amppred' }
    }
    kt_amppred_predictions {
        path { 'tools_predictions/kt_amppred' }
    }
    plapd_model {
        path { 'models/plapd' }
    }
    plapd_predictions {
        path { 'tools_predictions/plapd' }
    }
    dlfea4ampgen_predictions {
        path { 'tools_predictions/dlfea4ampgen' }
    }
    multiamp_predictions {
        path { 'tools_predictions/multiamp' }
    }
}