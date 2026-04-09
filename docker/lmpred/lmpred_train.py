# Adapted from 2_Build_CNN_Model.ipynb
# Source: https://github.com/williamdee1/LMPred_AMP_Prediction
# Changes: removed Google Colab dependencies, added argparse CLI interface.
# The model with the best reported performance (T5 trained on UniRef50) was selected

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import (Dense, Flatten, Dropout, Conv2D,
                          MaxPooling2D, BatchNormalization)
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt


# ─── Two-layer CNN model (original, T5XL-UNI hyperparameters) ─────────────────

def train_complex_model(X_train, y_train, X_val, y_val,
                        model_path, plots_path,
                        epochs, batch_size, use_tpu,
                        filter, k_size, k_init, pool, pool_strides,
                        filter2, k_size2, pool2, p_strides2,
                        dense1, dense2, dropout, lr, opt):

    def create_model():
        model = Sequential()

        model.add(Conv2D(filters=filter, kernel_size=k_size, activation='relu',
                         strides=1, kernel_initializer=k_init, padding='same',
                         input_shape=(255, 1024, 1)))
        model.add(MaxPooling2D(pool_size=pool, strides=pool_strides))
        model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))

        model.add(Conv2D(filters=filter2, kernel_size=k_size2, activation='relu',
                         strides=1, kernel_initializer=k_init, padding='same',
                         input_shape=(255, 1024, 1)))
        model.add(MaxPooling2D(pool_size=pool2, strides=p_strides2))
        model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))

        model.add(Flatten())
        model.add(Dense(dense1, activation='relu'))
        model.add(Dropout(rate=dropout))
        model.add(Dense(dense2, activation='relu'))
        model.add(Dropout(rate=dropout))
        model.add(Dense(1, activation='sigmoid'))

        if opt == 'Adam':
            optimizer = keras.optimizers.Adam(learning_rate=lr)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=lr)

        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if use_tpu:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        with strategy.scope():
            model = create_model()
    else:
        model = create_model()

    model.summary()

    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=12, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(
        model_path, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4,
        verbose=1, min_delta=1e-4, mode='min')

    seq_height = 255
    seq_width = 1024

    X_train_res = X_train.reshape(X_train.shape[0], seq_height, seq_width, 1)
    X_val_res = X_val.reshape(X_val.shape[0], seq_height, seq_width, 1)
    y_train_res = y_train.astype('float32').reshape((-1, 1))
    y_val_res = y_val.astype('float32').reshape((-1, 1))

    history = model.fit(
        X_train_res, y_train_res,
        validation_data=(X_val_res, y_val_res),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss]
    )

    # Training curves
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=80)
    train_metrics = ['accuracy', 'loss']
    val_metrics = ['val_accuracy', 'val_loss']
    titles = ['Model Accuracy', 'Model Loss vs. Learning Rate']
    y_labels = ['Accuracy', 'Loss']
    leg_loc = ['upper left', 'upper right']

    for i in range(2):
        ax[i].plot(history.history[train_metrics[i]])
        ax[i].plot(history.history[val_metrics[i]])

        if i == 1:
            ax2 = ax[i].twinx()
            ax2.plot(history.history['lr'], color='magenta', linestyle='dotted')
            ax2.set_ylabel('Learning Rate')
            ax2.legend(['Learning Rate'], fancybox=True, framealpha=1,
                       shadow=True, borderpad=1, bbox_to_anchor=(1.0, 0.85))
            ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))

        ax[i].set_title(titles[i], fontsize=12, fontweight='bold')
        ax[i].set_ylabel(y_labels[i])
        ax[i].set_xlabel('Epoch')
        ax[i].legend(['Train', 'Val'], loc=leg_loc[i], fancybox=True,
                     framealpha=1, shadow=True, borderpad=1)

    plt.savefig(plots_path, bbox_inches='tight')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LMPred two-layer CNN on T5XL-UNI embeddings."
    )
    parser.add_argument("--x_train_npy", required=True)
    parser.add_argument("--x_val_npy",   required=True)
    parser.add_argument("--y_train_csv", required=True)
    parser.add_argument("--y_val_csv",   required=True)
    parser.add_argument("--model_path",  required=True,
                        help="Path to save the best model (.keras).")
    parser.add_argument("--plots_path",  required=True,
                        help="Path to save training curves (.png).")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    X_train = np.load(args.x_train_npy)
    X_val   = np.load(args.x_val_npy)
    y_train = np.array(pd.read_csv(args.y_train_csv, header=None))
    y_val   = np.array(pd.read_csv(args.y_val_csv,   header=None))

    # T5XL-UNI best hyperparameters (original paper)
    train_complex_model(
        X_train, y_train, X_val, y_val,
        model_path=args.model_path,
        plots_path=args.plots_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_tpu=False,
        filter=352, k_size=27, k_init='RandomNormal',
        pool=2, pool_strides=4,
        filter2=128, k_size2=21,
        pool2=2, p_strides2=4,
        dense1=640, dense2=64,
        dropout=0.0, lr=0.0001, opt='Adam'
    )


if __name__ == "__main__":
    main()
