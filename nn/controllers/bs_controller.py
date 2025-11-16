import tensorflow as tf


class BSController(tf.keras.Model):
    def __init__(self, n_tx, n_cb, n_fb, use_learned_codebook=True):
        super().__init__()
        self.n_tx = n_tx
        self.n_cb = n_cb
        self.use_learned_codebook = use_learned_codebook

        # Learnable codebook (N3): real and imag parts
        if use_learned_codebook:
            self.codebook_real = tf.Variable(
                tf.random.normal([n_cb, n_tx], stddev=0.1), trainable=True, name="cb_real")
            self.codebook_imag = tf.Variable(
                tf.random.normal([n_cb, n_tx], stddev=0.1), trainable=True, name="cb_imag")
        else:
            # will be set externally with your DFT codebook
            self.codebook_real = None
            self.codebook_imag = None

        # N2: feedback -> final beam
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(2 * n_tx)  # real + imag
        ])

    def get_codebook(self):
        if self.use_learned_codebook:
            f_complex = tf.complex(self.codebook_real, self.codebook_imag)  # [N_cb, N_tx]
        else:
            f_complex = tf.complex(self.codebook_real, self.codebook_imag)
        # normalize rows
        norm = tf.sqrt(tf.reduce_sum(tf.abs(f_complex)**2, axis=-1, keepdims=True) + 1e-9)
        norm = tf.cast(norm, f_complex.dtype)
        return f_complex / norm  # [N_cb, N_tx]

    def get_fw_from_index(self, idx):
        """Return codebook beam f_t given index idx: [B], returns [B, n_tx] complex."""
        codebook = self.get_codebook()
        f_t = tf.gather(codebook, idx)  # [B, n_tx]
        return f_t

    def map_feedback_to_beam(self, m_FB):
        """Implements N2: m_FB -> f_T."""
        logits = self.ffn(m_FB)         # [B, 2*n_tx]
        f_real, f_imag = tf.split(logits, 2, axis=-1)
        f_T = tf.complex(f_real, f_imag)
        norm = tf.sqrt(tf.reduce_sum(tf.abs(f_T)**2, axis=-1, keepdims=True) + 1e-9)
        norm = tf.cast(norm, f_T.dtype)
        return f_T / norm
