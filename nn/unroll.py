import tensorflow as tf


def unroll_ba(H, ue_ctrl, bs_ctrl, T, snr_db=10.0):
    """
    Unroll the NN-based joint BA as in the paper (config C3).

    Args:
        H: [B, N_RX, N_TX] complex channels.
        ue_ctrl: UEController instance.
        bs_ctrl: BSController instance.
        T: number of sensing steps (UE receives T probes).
        snr_db: per-antenna SNR in dB.

    Returns:
        J: [B] final normalized beamforming gain.
    """
    B = tf.shape(H)[0]
    n_rx = H.shape[1]
    n_tx = H.shape[2]

    # Noise variance
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_lin

    # Initialize UE RNN state and previous values
    h_prev = tf.zeros([B, ue_ctrl.gru.units])
    y_prev = tf.zeros([B, 2])                  # (Re, Im) of y_{-1} = 0
    x_prev = -tf.ones([B], dtype=tf.int32)     # "no beam yet"
    w_prev = tf.zeros([B, 2 * n_rx])           # w_{-1} = 0

    # Random starting index i for BS codebook (not mandatory if you want simpler)
    i0 = tf.random.uniform([], minval=0, maxval=bs_ctrl.n_cb, dtype=tf.int32)

    # Sensing phase t = 0..T-1
    for t in range(T):
        # 1) UE: choose combiner w_t
        w_t, h_new = ue_ctrl.call_step(y_prev, x_prev, w_prev, h_prev)  # [B, n_rx]

        # 2) BS: choose codebook beam index and beam
        idx_t = (i0 + t) % bs_ctrl.n_cb
        idx_t = tf.fill([B], idx_t)   # same for all batch samples (can also randomize per-sample)
        f_t = bs_ctrl.get_fw_from_index(idx_t)  # [B, n_tx]

        # 3) Received pilot y_t = w^H H f + w^H n
        # H: [B, N_RX, N_TX]; w_t: [B, N_RX]; f_t: [B, N_TX]
        Hw = tf.einsum('brm,bm->br', H, f_t)          # H f_t: [B, N_RX]
        y_clean = tf.einsum('br,br->b', tf.math.conj(w_t), Hw)  # scalar per batch

        # Noise
        n = tf.complex(
            tf.random.normal(tf.shape(y_clean), stddev=tf.sqrt(sigma2/2.0)),
            tf.random.normal(tf.shape(y_clean), stddev=tf.sqrt(sigma2/2.0)),
        )
        y_t = y_clean + n

        # Prepare next-step inputs
        y_prev = tf.stack([tf.math.real(y_t), tf.math.imag(y_t)], axis=-1)  # [B, 2]
        x_prev = idx_t
        w_prev = tf.concat([tf.math.real(w_t), tf.math.imag(w_t)], axis=-1)
        h_prev = h_new

    # After T sensing steps: UE produces final combiner w_T and feedback m_FB
    w_T, m_FB = ue_ctrl.call_final(h_prev)  # [B, n_rx], [B, n_fb]

    # BS maps m_FB to final beam f_T
    f_T = bs_ctrl.map_feedback_to_beam(m_FB)  # [B, n_tx]

    # Final beamforming gain J = |w_T^H H f_T|^2 / ||H||_F^2
    Hf = tf.einsum('brm,bm->br', H, f_T)                 # [B, N_RX]
    y_final = tf.einsum('br,br->b', tf.math.conj(w_T), Hf)  # [B]
    num = tf.abs(y_final)**2

    H_norm2 = tf.reduce_sum(tf.abs(H)**2, axis=[1, 2]) + 1e-9
    J = num / H_norm2

    return J  # [B]
