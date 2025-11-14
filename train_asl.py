"""
Script de entrenamiento para el modelo ASL Fingerspelling.
"""

import os
import json
import pickle
import argparse
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from data_pipeline import (
    load_char_to_num_mapping,
    get_char_mappings,
    create_tfrecords,
    build_datasets,
    pad_token_idx,
    start_token_idx,
    end_token_idx,
    FRAME_LEN
)
from model_asl import build_model, cer_from_dense


class DisplayOutputs(keras.callbacks.Callback):
    """Callback para mostrar ejemplos de predicciones durante el entrenamiento."""
    
    def __init__(
        self,
        batch,
        idx_to_token,
        target_start_token_idx=60,
        target_end_token_idx=61,
        pad_idx=None,
        show_every=4
    ):
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token
        self.pad_idx = pad_idx
        self.show_every = show_every

    def on_epoch_end(self, epoch, logs=None):
        if self.show_every and (epoch % self.show_every != 0):
            return

        source = self.batch[0]
        target = self.batch[1].numpy()
        bs = tf.shape(source)[0]

        preds = self.model.generate(source, self.target_start_token_idx).numpy()

        remove_ids = [self.pad_idx, self.target_start_token_idx, self.target_end_token_idx]
        cer_val = float(cer_from_dense(preds, target, remove_ids).numpy()) if self.pad_idx is not None else float('nan')
        print(f"\n[DISPLAY BATCH] Epoch {epoch+1}: CER(batch) = {cer_val:.4f}\n")

        for i in range(min(3, int(bs))):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")


def train_model(
    csv_path,
    landmarks_path,
    char_mapping_path,
    output_dir="models",
    epochs=13,
    batch_size=64,
    create_records=False
):
    """
    Entrena el modelo ASL Fingerspelling.
    
    Args:
        csv_path: Ruta al archivo train.csv
        landmarks_path: Ruta a la carpeta con archivos parquet
        char_mapping_path: Ruta al archivo JSON con mapeo de caracteres
        output_dir: Carpeta para guardar el modelo
        epochs: Número de épocas
        batch_size: Tamaño del batch
        create_records: Si True, crea TFRecords desde cero
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar mapeo de caracteres
    char_to_num = load_char_to_num_mapping(char_mapping_path)
    char_to_num, num_to_char = get_char_mappings(char_to_num)
    
    # Crear TFRecords si es necesario
    if create_records:
        import pandas as pd
        dataset_df = pd.read_csv(csv_path)
        create_tfrecords(dataset_df, landmarks_path)
    
    # Obtener lista de TFRecords
    tf_records = [f"preprocessed/{f}" for f in os.listdir("preprocessed") if f.endswith(".tfrecord")]
    print(f"Found {len(tf_records)} TFRecord files.")
    
    # Construir datasets
    train_ds, valid_ds = build_datasets(tf_records, char_to_num, batch_size=batch_size)
    
    # Callback de visualización
    batch = next(iter(valid_ds))
    idx_to_char = list(char_to_num.keys())
    display_cb = DisplayOutputs(
        batch,
        idx_to_token=idx_to_char,
        target_start_token_idx=start_token_idx,
        target_end_token_idx=end_token_idx,
        pad_idx=pad_token_idx,
        show_every=4
    )
    
    # Construir modelo
    model = build_model(
        num_hid=200,
        num_head=4,
        num_feed_forward=400,
        source_maxlen=FRAME_LEN,
        target_maxlen=64,
        num_layers_enc=2,
        num_layers_dec=1,
        num_classes=62,
        learning_rate=0.0001,
        pad_token_idx=pad_token_idx,
        char_to_num=char_to_num
    )
    
    # Entrenar
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        callbacks=[display_cb],
        epochs=epochs
    )
    
    # Guardar modelo y artefactos
    model_path = os.path.join(output_dir, "asl_transformer")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Guardar pesos
    weights_path = os.path.join(output_dir, "asl_transformer_weights.h5")
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")
    
    # Guardar configuración y mapeos
    config = {
        'char_to_num': char_to_num,
        'num_to_char': num_to_char,
        'pad_token_idx': pad_token_idx,
        'start_token_idx': start_token_idx,
        'end_token_idx': end_token_idx,
        'num_classes': 62,
        'source_maxlen': FRAME_LEN,
        'target_maxlen': 64
    }
    
    config_path = os.path.join(output_dir, "config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"Configuration saved to {config_path}")
    
    # Guardar historia de entrenamiento
    history_path = os.path.join(output_dir, "history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {history_path}")
    
    # Graficar resultados
    plot_training_results(history.history, output_dir)
    
    return model, history


def plot_training_results(history, output_dir):
    """Genera gráficas de los resultados del entrenamiento."""
    import numpy as np
    
    epochs = np.arange(1, len(history['loss']) + 1)
    
    # Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], marker='o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], marker='o', label='Val Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Loss por época')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CER
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['edit_dist'], marker='o', label='Train CER')
    plt.plot(epochs, history['val_edit_dist'], marker='o', label='Val CER')
    plt.xlabel('Época')
    plt.ylabel('CER (↓ mejor)')
    plt.title('CER por época')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Brecha de generalización
    plt.subplot(1, 3, 3)
    gap_loss = np.array(history['val_loss']) - np.array(history['loss'])
    gap_cer = np.array(history['val_edit_dist']) - np.array(history['edit_dist'])
    plt.plot(epochs, gap_loss, marker='o', label='Gap Loss (val - train)')
    plt.plot(epochs, gap_cer, marker='o', label='Gap CER (val - train)')
    plt.axhline(0, color='gray', linewidth=1, linestyle='--')
    plt.xlabel('Época')
    plt.ylabel('Brecha')
    plt.title('Brecha de generalización')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
    print(f"Training plots saved to {os.path.join(output_dir, 'training_results.png')}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL Fingerspelling model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--landmarks_path", type=str, required=True, help="Path to landmarks directory")
    parser.add_argument("--char_mapping", type=str, required=True, help="Path to character mapping JSON")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=13, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--create_records", action="store_true", help="Create TFRecords from scratch")
    
    args = parser.parse_args()
    
    train_model(
        csv_path=args.csv_path,
        landmarks_path=args.landmarks_path,
        char_mapping_path=args.char_mapping,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        create_records=args.create_records
    )
