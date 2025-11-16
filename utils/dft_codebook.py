from typing import List
from sionna.phy.mimo.precoding import grid_of_beams_dft_ula
import tensorflow as tf

from constants import N_TX_ANT, OVERSAMPLING

def build_tx_codebook()-> List[tf.Tensor]:
    # gob: [N_beams, N_TX_ANT]
    gob = grid_of_beams_dft_ula(
        num_ant=N_TX_ANT,
        oversmpl=OVERSAMPLING,
    )
    print(f"TX DFT GoB codebook: {gob}")
    print(f"DFT codebook shape: {gob.shape}")
    return gob  # complex64 tensor
