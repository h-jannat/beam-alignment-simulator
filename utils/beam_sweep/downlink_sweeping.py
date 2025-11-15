import tensorflow as tf
from sionna.phy.channel.ofdm_channel import OFDMChannel

from constants import BATCH_SIZE, FFT_SIZE, N_TX_ANT


def sweep_beams_downlink(rg, channel:OFDMChannel , tx_codebook: tf.TensorArray , noise_variance=0.0):
    """
    Beam sweeping for downlink:
    - channel: OFDMChannel instance
    - tx_codebook: [N_beams, N_TX_ANT] complex tensor
    Returns:
        best_beam_idx: [B] int tensor
        per_beam_power: [B, N_beams] float tensor
    """
    N_beams = tx_codebook.shape[0]

    # Dummy pilot grid: all zeros is fine, we only want the channel
    #num_rx = 1 (we have one “link” / receiver node)
    # num_tx = 1 (one transmitter)
    # num_ofdm_symbols = 1
    x = tf.zeros(
        [BATCH_SIZE, 1, N_TX_ANT, rg.num_ofdm_symbols, rg.fft_size],
        dtype=tf.complex64,
    )

    # Make sure OFDMChannel returns the channel
    channel._return_channel = True  # public API has "return_channel" at init, this enforces it

    # Call channel: y, h_freq
    # 'no' is the noise variance used for AWGN; with 0 we get noise-free channel
    _, h_freq = channel(x, noise_variance)


#h_freq: [B, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]=
# [B, 1, N_RX_ANT, 1, N_TX_ANT, 1, FFT_SIZE]
# B – batch size (number of independent channel realizations)
# 1 – number of receivers (we only created one)
# N_RX_ANT – number of antennas at the UE
# 1 – number of transmitters (we only created one)
# N_TX_ANT – number of antennas at the BS
# 1 – number of OFDM symbols
# FFT_SIZE – number of subcarriers

    # Extract and reshape to [B, N_RX_ANT, FFT_SIZE, N_TX_ANT]
    H = h_freq[:, 0, :, 0, :, 0, :]             # [B, N_RX_ANT, N_TX_ANT, FFT_SIZE], since there is only 1 RX, 1 TX and 1 OFDM symbol
    # We want to reorder axes so that subcarrier dimension (FFT_SIZE) comes
    # before the TX antenna dimension, because later we want to align 
    # dimensions nicely with the codebook
    H = tf.transpose(H, [0, 1, 3, 2])           # [B, N_RX_ANT, FFT_SIZE, N_TX_ANT]

    # Add beam dimension and broadcast:
    # H_expanded: [B, 1, N_RX_ANT, FFT_SIZE, N_TX_ANT]
    H_expanded = tf.expand_dims(H, axis=1)

    # tx_codebook: [N_beams, N_TX_ANT] -> [1, N_beams, 1, 1, N_TX_ANT]
    gob_exp = tf.reshape(tx_codebook, [1, N_beams, 1, 1, N_TX_ANT])

    # Effective channel after TX beamforming:
    # sum over TX antennas -> [B, N_beams, N_RX_ANT, FFT_SIZE]
    # beamforming operation per subcarrier
    # Heff​[b,beam,r,k]=m=0∑NTX​−1 ​H[b,r,k,m]w[beam,m]
#     H is frequency-domain (per-subcarrier channel)
# w is spatial (per-antenna weights)
# We combine them to get an effective channel per beam and subcarrier
    H_eff = tf.reduce_sum(H_expanded * gob_exp, axis=-1)

    # Received power per beam: mean over RX antennas and subcarriers
    per_beam_power = tf.reduce_mean(tf.abs(H_eff) ** 2, axis=[2, 3])  # [B, N_beams]
    print(f"Per beam power shape: {per_beam_power.shape}")
    # Best beam index for each batch example
    best_beam_idx = tf.argmax(per_beam_power, axis=-1, output_type=tf.int32)

    return best_beam_idx, per_beam_power
