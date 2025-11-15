from sionna.phy.mimo.precoding import grid_of_beams_dft_ula

from constants import N_TX_ANT, OVERSAMPLING

def build_tx_codebook():
    # gob: [N_beams, N_TX_ANT]
    gob = grid_of_beams_dft_ula(
        num_ant=N_TX_ANT,
        oversmpl=OVERSAMPLING,
    )
    return gob  # complex64 tensor
