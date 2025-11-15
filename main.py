import tensorflow as tf
from constants import N_TX_ANT, OVERSAMPLING
from utils.beam_sweep.downlink_sweeping import sweep_beams_downlink
from utils.channel import build_cdl_channel, build_resource_grid
from utils.dft_codebook import build_tx_codebook


def main():
    tf.random.set_seed(0)

    # 1) resource grid
    rg = build_resource_grid()

    # 2) CDL-based OFDM channel (downlink)
    channel_dl = build_cdl_channel(rg, direction="downlink")

    # 3)  DFT GoB codebook for BS
    tx_codebook = build_tx_codebook()

    # 4) Run beam sweeping
    best_beam_idx, per_beam_power = sweep_beams_downlink(
        rg, channel_dl, tx_codebook, noise_variance=0.0
    )

    print("Best beam indices (first 10 realizations):")
    print(best_beam_idx.numpy()[:10])

    # Optionally: map beam index to azimuth angle (approx)
    # For ULA with N antennas and oversmpl=O, beam index m in [0, N*O-1]:
    # theta ≈ arcsin(2*m/N - 1) (approximation based on Sionna docs):contentReference[oaicite:5]{index=5}

    N_beams = tx_codebook.shape[0]
    m = tf.cast(best_beam_idx, tf.float32)
    N = float(N_TX_ANT * OVERSAMPLING)
    # crude mapping, valid for small angles; adjust as needed
    theta = tf.asin(tf.clip_by_value(2.0 * m / N - 1.0, -1.0, 1.0))
    print("Approx AoD (rad) for first 10 best beams:")
    print(theta.numpy()[:10])


if __name__ == "__main__":
    main()
