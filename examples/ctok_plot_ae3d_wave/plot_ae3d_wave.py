"""
This script shows how to select and plot a wave from AE3D output.
"""

# %% load dependencies ----------------------------------------------

# Plotly IO to save plots:
import plotly.io as pio

# Module to process outputs:
import stellgap as sg

# %% load AE3D output -----------------------------------------------

ae3d_output = sg.EigModeASCI('.')

# %% load AE3D mode closest to target frequency ---------------------

target_frequency_kHz = 46.083
tae = sg.AE3DEigenvector.from_eig_mode_asci(
    eig_mode_asci=ae3d_output,
    target_eigenvalue=target_frequency_kHz**2
)

# %% plot the loaded TAE and save to HTML & PNG --------------------

tae_fig = sg.plot_ae3d_eigenmode(
    mode=tae,
    harmonics=8
)

print("Saving overlaid continua plot... ")
pio.write_html(
    fig=tae_fig,
    file="ctok_ae3d_tae_mode.html",
    include_mathjax='cdn'
)
pio.write_image(
    fig=tae_fig,
    file="ctok_ae3d_tae_mode.png",
    format="png",
    scale=2
)
