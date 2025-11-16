import tensorflow as tf

from constants import N_CB, N_CB, N_RX_ANT, N_TX_ANT, OVERSAMPLING, T_STEPS, SNR_dB_train
from nn.controllers.bs_controller import BSController
from nn.controllers.ue_controller import UEController
from nn.unroll import unroll_ba
from utils import channel
from utils.dft_codebook import build_tx_codebook



ue_ctrl = UEController(n_rx=N_RX_ANT, n_cb=N_CB, n_fb=16)
bs_ctrl = BSController(n_tx=N_TX_ANT, n_cb=N_CB, n_fb=16, use_learned_codebook=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


#    If you want it FIXED (not trained), set trainable=False or leave it as a constant.
codebook_init = build_tx_codebook()      # [N_CB, N_TX_ANT], complex
codebook_tx = tf.Variable(codebook_init, trainable=True, name="tx_codebook")

@tf.function
def train_step():
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        
        # 1) Get channel from your existing simulator
        
        _, h_freq = channel()  # h_freq: [B, 1, N_RX_ANT, 1, N_TX_ANT, 1, FFT_SIZE]

        # Extract and reshape to [B, N_RX_ANT, FFT_SIZE, N_TX_ANT]
        H = h_freq[:, 0, :, 0, :, 0, :]          # [B, N_RX_ANT, N_TX_ANT, FFT_SIZE]
        H = tf.transpose(H, [0, 1, 3, 2])        # [B, N_RX_ANT, FFT_SIZE, N_TX_ANT]

        # Average over subcarriers -> narrowband H_nb: [B, N_RX_ANT, N_TX_ANT]
        H_nb = tf.reduce_mean(H, axis=2)

        
        # 2) Unroll BA with NNs (this should be differentiable)
        
        bf_gain_norm, logs = unroll_ba(
            H_nb,
            ue_ctrl,          # UEController instance (RNN/MLP at UE)
            bs_ctrl,          # BSController instance (MLP + codebook at BS)
            T=T_STEPS,
            snr_db=SNR_dB_train,
            CB_TX=codebook_tx  # learnable TX codebook variable
        )


        # Loss = negative expected normalized BF gain
        
        loss = -tf.reduce_mean(bf_gain_norm)

    
    # 4) Backprop: compute gradients and apply update
    
    vars_all = (
        ue_ctrl.trainable_variables
        + bs_ctrl.trainable_variables
        + [codebook_tx]  # remove this if you want fixed (non-learnable) codebook
    )

    grads = tape.gradient(loss, vars_all)
    optimizer.apply_gradients(zip(grads, vars_all))

    return loss, tf.reduce_mean(bf_gain_norm)
