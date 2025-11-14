"""
Modelo Transformer para ASL Fingerspelling.
Contiene la arquitectura del modelo y funciones para construcción y compilación.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TokenEmbedding(layers.Layer):
    """Embedding para tokens de texto con codificación posicional."""
    
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class LandmarkEmbedding(layers.Layer):
    """Embedding para landmarks usando capas convolucionales."""
    
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv3 = layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class TransformerEncoder(layers.Layer):
    """Bloque encoder del Transformer."""
    
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoder(layers.Layer):
    """Bloque decoder del Transformer."""
    
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Máscara causal para prevenir atención a tokens futuros."""
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat([batch_size[..., tf.newaxis], tf.constant([1, 1], dtype=tf.int32)], 0)
        return tf.tile(mask, mult)

    def call(self, inputs, training=None):
        enc_out, target = inputs

        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att, training=training))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out, training=training) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training=training))
        return ffn_out_norm


def dense_to_sparse_without_ids(dense_ids, remove_ids):
    """Convierte tensor denso a SparseTensor filtrando IDs específicos."""
    dense_ids = tf.convert_to_tensor(dense_ids)
    mask = tf.ones_like(dense_ids, dtype=tf.bool)
    for rid in remove_ids:
        if rid is not None:
            mask = tf.logical_and(mask, tf.not_equal(dense_ids, rid))
    indices = tf.where(mask)
    values = tf.gather_nd(dense_ids, indices)
    dense_shape = tf.cast(tf.shape(dense_ids), tf.int64)
    return tf.SparseTensor(
        indices=tf.cast(indices, tf.int64),
        values=tf.cast(values, tf.int32),
        dense_shape=dense_shape
    )


def cer_from_dense(pred_tokens, true_tokens, remove_ids):
    """Calcula CER (Character Error Rate) promedio."""
    with tf.device('/CPU:0'):
        pred_sp = dense_to_sparse_without_ids(pred_tokens, remove_ids)
        true_sp = dense_to_sparse_without_ids(true_tokens, remove_ids)
        dist = tf.edit_distance(pred_sp, true_sp, normalize=True)
        return tf.reduce_mean(dist)


class Transformer(keras.Model):
    """Modelo Transformer completo para ASL Fingerspelling."""
    
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=60,
        pad_token_idx=59,
        char_to_num=None
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.Mean(name="edit_dist")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        self.pad_token_idx = pad_token_idx
        self.char_to_num = char_to_num or {}

        self.enc_input = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input] + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target, training=None):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")([enc_out, y], training=training)
        return y

    def call(self, inputs, training=None):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source, training=training)
        y = self.decode(x, target, training=training)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        source, target = batch
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        with tf.GradientTape() as tape:
            preds = self([source, dec_input], training=True)
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, self.pad_token_idx))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        pred_tokens = tf.cast(tf.argmax(preds, axis=-1), tf.int32)
        truth_tokens = dec_target

        remove_ids = [self.pad_token_idx, self.char_to_num.get('<'), self.char_to_num.get('>')]
        cer = cer_from_dense(pred_tokens, truth_tokens, remove_ids)

        self.acc_metric.update_state(cer)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result(), "edit_dist": self.acc_metric.result()}

    def test_step(self, batch):
        source, target = batch
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]

        preds = self([source, dec_input], training=False)
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, self.pad_token_idx))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)

        pred_tokens = tf.cast(tf.argmax(preds, axis=-1), tf.int32)
        truth_tokens = dec_target

        remove_ids = [self.pad_token_idx, self.char_to_num.get('<'), self.char_to_num.get('>')]
        cer = cer_from_dense(pred_tokens, truth_tokens, remove_ids)

        self.acc_metric.update_state(cer)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result(), "edit_dist": self.acc_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Genera predicciones usando greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source, training=False)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input, training=False)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input


def build_model(
    num_hid=200,
    num_head=4,
    num_feed_forward=400,
    source_maxlen=128,
    target_maxlen=64,
    num_layers_enc=2,
    num_layers_dec=1,
    num_classes=62,
    learning_rate=0.0001,
    pad_token_idx=59,
    char_to_num=None
):
    """
    Construye y compila el modelo Transformer.
    
    Args:
        num_hid: Dimensión del espacio de embeddings
        num_head: Número de cabezas de atención
        num_feed_forward: Dimensión de la capa feed-forward
        source_maxlen: Longitud máxima de la secuencia de entrada
        target_maxlen: Longitud máxima de la secuencia objetivo
        num_layers_enc: Número de capas del encoder
        num_layers_dec: Número de capas del decoder
        num_classes: Número de clases (caracteres)
        learning_rate: Tasa de aprendizaje
        pad_token_idx: Índice del token de padding
        char_to_num: Diccionario de mapeo de caracteres
        
    Returns:
        Modelo compilado
    """
    model = Transformer(
        num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        source_maxlen=source_maxlen,
        target_maxlen=target_maxlen,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=num_classes,
        pad_token_idx=pad_token_idx,
        char_to_num=char_to_num
    )
    
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=0.1,
    )
    
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, jit_compile=False)
    
    return model
