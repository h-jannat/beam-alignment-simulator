import tensorflow as tf
import numpy as np

from constants import (
    N_RX_ANT,
    N_TX_ANT,
    OVERSAMPLING,
    T_STEPS,
    SNR_dB_test,      # e.g. list or scalar
)
from nn.controllers.ue_controller import UEController
from nn.controllers.bs_controller import BSController
from nn.unroll import unroll_ba
from utils.channel import channel
from utils.dft_codebook import build_tx_codebook
from utils.beam_sweep import sweep_beams_downlink   # whatever file holds your baseline sweep

N_CB = N_TX_ANT * OVERSAMPLING

ue_ctrl = UEController(n_rx=N_RX_ANT, n_cb=N_CB, n_fb=16)
bs_ctrl = BSController(n_tx=N_TX_ANT, n_cb=N_CB, n_fb=16, use_learned_codebook=True)

codebook_tx   = tf.Variable(build_tx_codebook(), trainable=True, name="tx_codebook")
optimizer     = tf.keras.optimizers.Adam(1e-3)  # needed for checkpoint

ckpt = tf.train.Checkpoint(
    ue_ctrl=ue_ctrl,
    bs_ctrl=bs_ctrl,
    codebook_tx=codebook_tx,
    optimizer=optimizer,
)
manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=5)
ckpt.restore(manager.latest_checkpoint).expect_partial()
print("Restored from", manager.latest_checkpoint)

@tf.function
def eval_batch():
    _, h_freq = channel()
    H = h_freq[:, 0, :, 0, :, 0, :]
    H = tf.transpose(H, [0, 1, 3, 2])
    H_nb = tf.reduce_mean(H, axis=2)

    # NN-based BA
    bf_gain_nn = unroll_ba(
        H_nb, ue_ctrl, bs_ctrl,
        T=T_STEPS,
        snr_db=SNR_dB_test,
        CB_TX=codebook_tx,
    )

    # Classical exhaustive beam sweep baseline, using same codebook
    bf_gain_sweep = sweep_beams_downlink(H_nb, CB_TX=codebook_tx)

    return bf_gain_nn, bf_gain_sweep

def main():
    NUM_BATCHES = 200

    gains_nn = []
    gains_sw = []

    for _ in range(NUM_BATCHES):
        g_nn, g_sw = eval_batch()
        gains_nn.append(g_nn.numpy())
        gains_sw.append(g_sw.numpy())

    gains_nn = np.concatenate(gains_nn, axis=0)
    gains_sw = np.concatenate(gains_sw, axis=0)

    print("Mean BF gain (NN):     ", gains_nn.mean())
    print("Mean BF gain (sweep):  ", gains_sw.mean())

if __name__ == "__main__":
    main()
