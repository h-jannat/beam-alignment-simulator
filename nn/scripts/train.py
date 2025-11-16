# nn/scripts/train.py

import tensorflow as tf
from constants import (
    BATCH_SIZE, N_RX_ANT, N_TX_ANT, OVERSAMPLING,
    T_STEPS, SNR_dB_train,
)
from nn.controllers.ue_controller import UEController
from nn.controllers.bs_controller import BSController
from nn.unroll import unroll_ba
from utils.channel import build_cdl_channel, build_resource_grid
from utils.dft_codebook import build_tx_codebook

N_CB = N_TX_ANT * OVERSAMPLING

# Models
ue_ctrl = UEController(n_rx=N_RX_ANT, n_cb=N_CB, n_fb=16)
bs_ctrl = BSController(n_tx=N_TX_ANT, n_cb=N_CB, n_fb=16,
                       use_learned_codebook=True)

# Learnable TX codebook
codebook_init = build_tx_codebook()              # [N_CB, N_TX_ANT]
codebook_tx   = tf.Variable(codebook_init,
                            trainable=True,
                            name="tx_codebook")

optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
         # 1) resource grid
        rg = build_resource_grid()

    # 2) CDL-based OFDM channel (downlink)
        channel_dl = build_cdl_channel(rg, direction="downlink")
        x = tf.zeros(
        [BATCH_SIZE, 1, N_TX_ANT, rg.num_ofdm_symbols, rg.fft_size],
        dtype=tf.complex64,
    )

    # Make sure OFDMChannel returns the channel
        channel_dl._return_channel = True  # public API has "return_channel" at init, this enforces it

    # Call channel: y, h_freq
    # 'no' is the noise variance used for AWGN; with 0 we get noise-free channel
        _, h_freq = channel_dl(x, noise_variance=0.0)
 

        # 2) Narrowband equivalent H_nb: [B, N_RX_ANT, N_TX_ANT]
        H = h_freq[:, 0, :, 0, :, 0, :]          # [B, N_RX, N_TX, FFT]
        H = tf.transpose(H, [0, 1, 3, 2])        # [B, N_RX, FFT, N_TX]
        H_nb = tf.reduce_mean(H, axis=2)         # [B, N_RX, N_TX]

        # 3) Unrolled beam alignment (NN controllers + codebook)
        bf_gain_norm, logs = unroll_ba(
            H_nb,
            ue_ctrl,
            bs_ctrl,
            T=T_STEPS,
            snr_db=SNR_dB_train,
            CB_TX=codebook_tx,
        )

        # 4) Loss = -E[normalized BF gain]
        loss = -tf.reduce_mean(bf_gain_norm)

    vars_all = (
        ue_ctrl.trainable_variables
        + bs_ctrl.trainable_variables
        + [codebook_tx]
    )
    grads = tape.gradient(loss, vars_all)
    optimizer.apply_gradients(zip(grads, vars_all))

    return loss, tf.reduce_mean(bf_gain_norm)

def main():
    NUM_UPDATES = 10_000
    LOG_EVERY   = 100

    ckpt = tf.train.Checkpoint(
        ue_ctrl=ue_ctrl,
        bs_ctrl=bs_ctrl,
        codebook_tx=codebook_tx,
        optimizer=optimizer,
    )
    manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=5)

    for step in range(1, NUM_UPDATES + 1):
        loss, gain = train_step()

        if step % LOG_EVERY == 0:
            tf.print("step:", step, "loss:", loss, "mean_gain:", gain)

        if step % (10 * LOG_EVERY) == 0:
            manager.save()
            tf.print("checkpoint saved at step", step)

if __name__ == "__main__":
    main()
