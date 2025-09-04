"""
This script shows how to contextualize AE3D data w.r.t. STELLGAP continuum.
"""

# %% load dependencies ----------------------------------------------

# Plotly IO to save plots:
import plotly.io as pio

# Module to process outputs:
import stellgap as sg

# %% load STELLGAP and AE3D outputs ---------------------------------

stellgap_output = sg.AlfvenSpecData.from_dir('.')
stellgap_continuum = stellgap_output.get_modes()
ae3d_output = sg.EigModeASCI('.')

# %% compute eigenvector maxima placements from AE3D ----------------

print("Computing continuum approximation from AE3d data...")
ae3d_continuum_approximation = sg.continuum_from_ae3d(
    ae3d_output,
    minevalue=0.0, 
    maxevalue=600**2
)

# %% overlay STELLGAP continuum with AE3D eigenvector resonances ----
#    and save results as HTML and PNG images

overlay_fig = sg.plot_continua(
    overlays=[
        stellgap_continuum,
        ae3d_continuum_approximation
    ]
)

print("Saving overlaid continua plot... ")
pio.write_html(
    fig=overlay_fig,
    file="ctok_stellgap_ae3d_continuum_overlay.html",
    include_mathjax='cdn'
)
pio.write_image(
    fig=overlay_fig,
    file="ctok_stellgap_ae3d_continuum_overlay.png",
    format="png",
    scale=2
)
