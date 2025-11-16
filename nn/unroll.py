import tensorflow as tf


def unroll_ba(H, ue_ctrl, bs_ctrl, T, snr_db=10.0, CB_TX=None):
    """
    Unroll the NN-based joint BA as in the paper (config C3).

    Args:
        H: [B, N_RX, N_TX] complex channels.
        ue_ctrl: UEController instance.
        bs_ctrl: BSController instance.
        T: number of sensing steps (UE receives T probes).
        snr_db: per-antenna SNR in dB.
        CB_TX: Optional complex tensor [N_cb, N_TX] representing a learnable TX
               codebook to use during the sensing phase. When None, the BS
               controller's internal codebook is used.

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
    h_prev = tf.ensure_shape(h_prev, [None, ue_ctrl.gru.units])
    y_prev = tf.zeros([B, 2])                  # (Re, Im) of y_{-1} = 0
    y_prev = tf.ensure_shape(y_prev, [None, 2])
    x_prev = -tf.ones([B], dtype=tf.int32)     # "no beam yet"
    x_prev = tf.ensure_shape(x_prev, [None])
    w_prev = tf.zeros([B, 2 * n_rx])           # w_{-1} = 0
    w_prev = tf.ensure_shape(w_prev, [None, 2 * n_rx])

    # Optional external TX codebook (unit-norm per beam)
    if CB_TX is not None:
        cb_norm = tf.sqrt(tf.reduce_sum(tf.abs(CB_TX)**2, axis=-1, keepdims=True) + 1e-9)
        cb_norm = tf.cast(cb_norm, CB_TX.dtype)
        cb_tx_norm = CB_TX / cb_norm
    else:
        cb_tx_norm = None

    # Random starting index i for BS codebook (not mandatory if you want simpler)
    i0 = tf.random.uniform([], minval=0, maxval=bs_ctrl.n_cb, dtype=tf.int32)
    T_int = tf.cast(T, tf.int32)

    def loop_cond(t, *_):
        return t < T_int

    def loop_body(t, h_prev, y_prev, x_prev, w_prev):
        w_t, h_new = ue_ctrl.call_step(y_prev, x_prev, w_prev, h_prev)  # [B, n_rx]

        idx_scalar = (i0 + t) % bs_ctrl.n_cb
        idx_t = tf.ones_like(x_prev) * idx_scalar
        if cb_tx_norm is not None:
            f_t = tf.gather(cb_tx_norm, idx_t)  # [B, n_tx]
        else:
            f_t = bs_ctrl.get_fw_from_index(idx_t)  # [B, n_tx]

        Hw = tf.einsum('brm,bm->br', H, f_t)          # H f_t: [B, N_RX]
        y_clean = tf.einsum('br,br->b', tf.math.conj(w_t), Hw)

        n = tf.complex(
            tf.random.normal(tf.shape(y_clean), stddev=tf.sqrt(sigma2/2.0)),
            tf.random.normal(tf.shape(y_clean), stddev=tf.sqrt(sigma2/2.0)),
        )
        y_t = y_clean + n

        y_prev_next = tf.stack([tf.math.real(y_t), tf.math.imag(y_t)], axis=-1)
        w_prev_next = tf.concat([tf.math.real(w_t), tf.math.imag(w_t)], axis=-1)
        return t + 1, h_new, y_prev_next, idx_t, w_prev_next

    _, h_prev, y_prev, x_prev, w_prev = tf.while_loop(
        loop_cond,
        loop_body,
        (tf.constant(0, dtype=tf.int32), h_prev, y_prev, x_prev, w_prev),
        parallel_iterations=1,
        maximum_iterations=T_int,
    )

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
