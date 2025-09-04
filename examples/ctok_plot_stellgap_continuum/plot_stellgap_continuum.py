"""
This script shows how to load and plot STELLGAP continuum output.
"""

# %% load dependencies ------------------------------------

# Plotly IO to save plots:
import plotly.io as pio

# Module to process outputs:
import stellgap as sg

# %% load STELLGAP output ---------------------------------

stellgap_output = sg.AlfvenSpecData.from_dir('.')

# %% plot continuum into png abd html files ---------------

continuum_fig = sg.plot_continuum(stellgap_output.get_modes())
pio.write_html(
    fig=continuum_fig,
    file="ctok_stellgap.html",
    include_mathjax='cdn'
)

pio.write_image(
    fig=continuum_fig,
    file="ctok_stellgap.png",
    format="png",
    scale=2
)
