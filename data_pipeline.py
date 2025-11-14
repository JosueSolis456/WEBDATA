"""
Data pipeline para ASL Fingerspelling.
Contiene funciones para cargar, preprocesar y construir datasets.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from tqdm import tqdm

# Configuración de landmarks
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

FEATURE_COLUMNS = X + Y + Z

X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in LPOSE]

# Longitud de frames
FRAME_LEN = 128

# Tokens especiales
pad_token = 'P'
start_token = '<'
end_token = '>'

pad_token_idx = 59
start_token_idx = 60
end_token_idx = 61


def load_char_to_num_mapping(json_path):
    """Carga el mapeo de caracteres a índices desde un archivo JSON."""
    with open(json_path, "r") as f:
        char_to_num = json.load(f)
    
    # Agregar tokens especiales
    char_to_num[pad_token] = pad_token_idx
    char_to_num[start_token] = start_token_idx
    char_to_num[end_token] = end_token_idx
    
    return char_to_num


def get_char_mappings(char_to_num):
    """Retorna el mapeo bidireccional de caracteres."""
    num_to_char = {j: i for i, j in char_to_num.items()}
    return char_to_num, num_to_char


def ajustar_rellenar(x):
    """Ajusta y rellena los frames a longitud FRAME_LEN."""
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN - tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x


def pre_process(x):
    """
    Preprocesa los landmarks:
    - Detecta mano dominante (la que tiene menos NaNs)
    - Normaliza coordenadas
    - Ajusta a FRAME_LEN frames
    """
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)

    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)

    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)

    # Seleccionar mano dominante
    if rnans > lnans:
        hand = lhand
        pose = lpose

        hand_x = hand[:, 0*(len(LHAND_IDX)//3): 1*(len(LHAND_IDX)//3)]
        hand_y = hand[:, 1*(len(LHAND_IDX)//3): 2*(len(LHAND_IDX)//3)]
        hand_z = hand[:, 2*(len(LHAND_IDX)//3): 3*(len(LHAND_IDX)//3)]
        hand = tf.concat([1-hand_x, hand_y, hand_z], axis=1)

        pose_x = pose[:, 0*(len(LPOSE_IDX)//3): 1*(len(LPOSE_IDX)//3)]
        pose_y = pose[:, 1*(len(LPOSE_IDX)//3): 2*(len(LPOSE_IDX)//3)]
        pose_z = pose[:, 2*(len(LPOSE_IDX)//3): 3*(len(LPOSE_IDX)//3)]
        pose = tf.concat([1-pose_x, pose_y, pose_z], axis=1)
    else:
        hand = rhand
        pose = rpose

    hand_x = hand[:, 0*(len(LHAND_IDX)//3): 1*(len(LHAND_IDX)//3)]
    hand_y = hand[:, 1*(len(LHAND_IDX)//3): 2*(len(LHAND_IDX)//3)]
    hand_z = hand[:, 2*(len(LHAND_IDX)//3): 3*(len(LHAND_IDX)//3)]
    hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)

    # Normalización
    mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    hand = (hand - mean) / std

    pose_x = pose[:, 0*(len(LPOSE_IDX)//3): 1*(len(LPOSE_IDX)//3)]
    pose_y = pose[:, 1*(len(LPOSE_IDX)//3): 2*(len(LPOSE_IDX)//3)]
    pose_z = pose[:, 2*(len(LPOSE_IDX)//3): 3*(len(LPOSE_IDX)//3)]
    pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]], axis=-1)

    x = tf.concat([hand, pose], axis=1)
    x = ajustar_rellenar(x)

    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    return x


def create_tfrecords(dataset_df, landmarks_path, output_dir="preprocessed"):
    """
    Crea TFRecords a partir del dataset original.
    
    Args:
        dataset_df: DataFrame con train.csv
        landmarks_path: Ruta a la carpeta con archivos parquet
        output_dir: Carpeta de salida para los TFRecords
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    for file_id in tqdm(dataset_df.file_id.unique(), desc="Creating TFRecords"):
        file_df = dataset_df.loc[dataset_df["file_id"] == file_id]
        parquet_df = pq.read_table(
            f"{landmarks_path}/{str(file_id)}.parquet",
            columns=['sequence_id'] + FEATURE_COLUMNS
        ).to_pandas()
        
        tf_file = f"{output_dir}/{file_id}.tfrecord"
        parquet_numpy = parquet_df.to_numpy()
        
        with tf.io.TFRecordWriter(tf_file) as file_writer:
            for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
                frames = parquet_numpy[parquet_df.index == seq_id]

                r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis=1) == 0)
                l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis=1) == 0)
                no_nan = max(r_nonan, l_nonan)

                if 2 * len(phrase) < no_nan:
                    features = {
                        FEATURE_COLUMNS[i]: tf.train.Feature(
                            float_list=tf.train.FloatList(value=frames[:, i])
                        ) for i in range(len(FEATURE_COLUMNS))
                    }
                    features["phrase"] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')])
                    )
                    record_bytes = tf.train.Example(
                        features=tf.train.Features(feature=features)
                    ).SerializeToString()
                    file_writer.write(record_bytes)


def decode_fn(record_bytes):
    """Decodifica un registro TFRecord."""
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in FEATURE_COLUMNS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
    features = tf.io.parse_single_example(record_bytes, schema)
    phrase = features["phrase"]
    landmarks = [tf.sparse.to_dense(features[COL]) for COL in FEATURE_COLUMNS]
    landmarks = tf.transpose(landmarks)
    return landmarks, phrase


def convert_fn(landmarks, phrase, table):
    """
    Convierte landmarks y frase a formato de entrenamiento.
    
    Args:
        landmarks: Tensor con coordenadas de landmarks
        phrase: Frase objetivo (string)
        table: Tabla de lookup para convertir caracteres a índices
    """
    # Agregar tokens de inicio y fin
    phrase = start_token + phrase + end_token
    phrase = tf.strings.bytes_split(phrase)
    phrase = table.lookup(phrase)
    # Padding
    phrase = tf.pad(
        phrase, 
        paddings=[[0, 64 - tf.shape(phrase)[0]]], 
        mode='CONSTANT',
        constant_values=pad_token_idx
    )
    return pre_process(landmarks), phrase


def build_datasets(tf_records, char_to_num, batch_size=64, train_split=0.8):
    """
    Construye datasets de entrenamiento y validación.
    
    Args:
        tf_records: Lista de archivos TFRecord
        char_to_num: Diccionario de mapeo de caracteres a índices
        batch_size: Tamaño del batch
        train_split: Proporción de datos para entrenamiento
        
    Returns:
        train_ds, valid_ds: Datasets de TensorFlow
    """
    # Crear tabla de lookup
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=list(char_to_num.keys()),
            values=list(char_to_num.values()),
        ),
        default_value=tf.constant(-1),
        name="class_weight"
    )
    
    train_len = int(train_split * len(tf_records))
    
    train_ds = (
        tf.data.TFRecordDataset(tf_records[:train_len])
        .map(decode_fn)
        .map(lambda x, y: convert_fn(x, y, table))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .cache()
    )
    
    valid_ds = (
        tf.data.TFRecordDataset(tf_records[train_len:])
        .map(decode_fn)
        .map(lambda x, y: convert_fn(x, y, table))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .cache()
    )
    
    return train_ds, valid_ds


def load_dataset_for_analysis(csv_path, landmarks_path, sample_size=1000):
    """
    Carga y preprocesa datos para análisis exploratorio (dashboard).
    
    Args:
        csv_path: Ruta al archivo train.csv
        landmarks_path: Ruta a la carpeta con archivos parquet
        sample_size: Número máximo de secuencias a cargar
        
    Returns:
        DataFrame con métricas agregadas para visualización
    """
    dataset_df = pd.read_csv(csv_path)
    
    # Limitar el tamaño para análisis
    if len(dataset_df) > sample_size:
        dataset_df = dataset_df.sample(n=sample_size, random_state=42)
    
    analysis_data = []
    
    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Loading data for analysis"):
        sequence_id = row['sequence_id']
        file_id = row['file_id']
        phrase = row['phrase']
        
        try:
            seq_df = pq.read_table(
                f"{landmarks_path}/{str(file_id)}.parquet",
                filters=[[('sequence_id', '=', sequence_id)]],
                columns=['sequence_id'] + FEATURE_COLUMNS[:20]  # Solo primeras columnas para velocidad
            ).to_pandas()
            
            num_frames = len(seq_df)
            phrase_length = len(phrase)
            
            # Calcular métricas agregadas
            right_hand_cols = [col for col in seq_df.columns if 'x_right_hand' in col]
            left_hand_cols = [col for col in seq_df.columns if 'x_left_hand' in col]
            
            right_hand_var = seq_df[right_hand_cols].var().mean() if right_hand_cols else 0
            left_hand_var = seq_df[left_hand_cols].var().mean() if left_hand_cols else 0
            
            analysis_data.append({
                'sequence_id': sequence_id,
                'file_id': file_id,
                'phrase': phrase,
                'phrase_length': phrase_length,
                'num_frames': num_frames,
                'frames_per_char': num_frames / phrase_length if phrase_length > 0 else 0,
                'right_hand_variance': right_hand_var,
                'left_hand_variance': left_hand_var,
                'dominant_hand': 'right' if right_hand_var > left_hand_var else 'left'
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(analysis_data)
