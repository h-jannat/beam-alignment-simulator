import tensorflow as tf
import numpy as np

# Sionna imports (module paths are explicit to avoid ambiguity)
from sionna.phy.ofdm.resource_grid import ResourceGrid
from sionna.phy.channel.ofdm_channel import OFDMChannel
from sionna.phy.channel.tr38901.cdl import CDL
from sionna.phy.channel.tr38901.antenna import PanelArray

from constants import (CARRIER_FREQUENCY, DELAY_SPREAD,
                       FFT_SIZE, N_SYM, SCS,
                       BS_NUM_ROWS, BS_NUM_COLS, BS_POLARIZATION,
                       UT_NUM_ROWS, UT_NUM_COLS)


def build_resource_grid():
    rg = ResourceGrid(
        num_ofdm_symbols=N_SYM,
        fft_size=FFT_SIZE,
        subcarrier_spacing=SCS,
        num_tx=1,                 # one BS
        num_streams_per_tx=1,     # single stream (SSB-like pilot)
        cyclic_prefix_length=0,
        num_guard_carriers=(0, 0),
        dc_null=False,
        pilot_pattern=None,       # empty pilot pattern
        pilot_ofdm_symbol_indices=None,
    )
    return rg


def build_cdl_channel(rg, direction="downlink"):
    # gNB panel (4x4 dual-pol, 38.901 pattern)
    bs_array = PanelArray(
        num_rows_per_panel=BS_NUM_ROWS,
        num_cols_per_panel=BS_NUM_COLS,
        polarization=BS_POLARIZATION,   # "dual"
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=CARRIER_FREQUENCY,
    )

    # UE: single-antenna, omni pattern
    ut_array = PanelArray(
        num_rows_per_panel=UT_NUM_ROWS,
        num_cols_per_panel=UT_NUM_COLS,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQUENCY,
    )

    cdl = CDL(
        model="A",                     # CDL-A
        delay_spread=DELAY_SPREAD,
        carrier_frequency=CARRIER_FREQUENCY,
        ut_array=ut_array,
        bs_array=bs_array,
        direction=direction,          # "downlink" or "uplink"
    )

    channel = OFDMChannel(
        channel_model=cdl,
        resource_grid=rg
        # Other args (add_awgn, normalize_channel, return_channel, etc.) are left at defaults
    )

    return channel
