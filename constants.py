# ----- System parameters -----

# BATCH_SIZE:  Monte-Carlo realizations (channel drops),
#  different random small-scale fading realizations (different path phases, Doppler, etc.)
import tensorflow as tf


BATCH_SIZE = 64          
CARRIER_FREQUENCY = 28e9 # 28 GHz mmWave
DELAY_SPREAD = 100e-9    # 100 ns, e.g. nominal UMi profile

FFT_SIZE = 64            # OFDM FFT size
N_SYM = 1                # Number of OFDM symbols (think: one SSB symbol),  1 for simplicity
SCS = 120e3              # Subcarrier spacing in Hz

# Antenna config: gNB has 4x4 dual-pol panel => 32 elements
BS_NUM_ROWS = 4
BS_NUM_COLS = 4
BS_POLARIZATION = "dual"   # 2 pols
N_TX_ANT = BS_NUM_ROWS * BS_NUM_COLS * 2

# UE: single-antenna for simplicity
UT_NUM_ROWS = 1
UT_NUM_COLS = 1
N_RX_ANT = 1

# DFT codebook oversampling in angle domain
OVERSAMPLING = 1  # 1 => N_TX_ANT beams, can increase if you want finer grid
T_STEPS = 8      # Number of sensing steps (probes)
# OVERSAMPLING = 4
N_CB = N_TX_ANT * OVERSAMPLING  # codebook size
# For training :
SNR_dB_train = 10.0      # per-antenna SNR

# For testing /:
SNR_dB_test = tf.constant([-10., 0., 5., 10., 15., 20.], dtype=tf.float32)
