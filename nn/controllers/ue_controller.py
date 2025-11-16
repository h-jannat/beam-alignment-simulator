import tensorflow as tf


class UEController(tf.keras.Model):
    def __init__(self, n_rx, n_cb, n_fb, rnn_units=128):
        super().__init__()
        self.n_rx = n_rx
        self.n_cb = n_cb
        self.n_fb = n_fb

        # Simple embedding for BS beam index
        self.emb = tf.keras.layers.Embedding(input_dim=n_cb+1, output_dim=16)  # +1 for "no beam yet" at t=0

        # GRU for temporal state
        self.gru = tf.keras.layers.GRU(rnn_units, return_state=True, return_sequences=False)

        # Dense heads for combiner and feedback
        self.dense_w = tf.keras.layers.Dense(2 * n_rx)      # real + imag
        self.dense_fb = tf.keras.layers.Dense(n_fb)         # real feedback vector

    def call_step(self, y_prev, x_prev, w_prev, h_prev):
        """
        y_prev: [B, 2]   (Re/Im)
        x_prev: [B]      (beam index, int, -1 for init)
        w_prev: [B, 2*n_rx]
        h_prev: [B, rnn_units]  (GRU state)
        """
        # Embed beam index; shift by +1 so -1 -> 0 works
        x_embed = self.emb(tf.clip_by_value(x_prev + 1, 0, self.n_cb))

        # Concatenate all inputs
        inp = tf.concat([y_prev, w_prev, x_embed], axis=-1)  # [B, D]

        # GRU step (we cheat: GRU normally expects time axis; here we treat each step separately)
        out, h_new = self.gru(tf.expand_dims(inp, axis=1), initial_state=h_prev)

        # Combiner head
        w_logits = self.dense_w(out)             # [B, 2*n_rx]
        w_real, w_imag = tf.split(w_logits, 2, axis=-1)
        w_complex = tf.complex(w_real, w_imag)   # [B, n_rx]

        # Normalize to unit norm
        w_norm = tf.sqrt(tf.reduce_sum(tf.abs(w_complex)**2, axis=-1, keepdims=True) + 1e-9)
        w_complex = w_complex / w_norm

        return w_complex, h_new

    def call_final(self, h_T):
        """From final hidden state produce w_T and m_FB."""
        out = h_T
        w_logits = self.dense_w(out)
        w_real, w_imag = tf.split(w_logits, 2, axis=-1)
        w_T = tf.complex(w_real, w_imag)
        w_norm = tf.sqrt(tf.reduce_sum(tf.abs(w_T)**2, axis=-1, keepdims=True) + 1e-9)
        w_T = w_T / w_norm

        m_FB = self.dense_fb(out)   # [B, n_fb], real

        return w_T, m_FB
