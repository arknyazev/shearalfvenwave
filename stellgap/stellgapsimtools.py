import os
import numpy as np
import plotly.graph_objects as go

from continuum import PlasmaDat
from continuum import FourierDat
from continuum import AlfvenSpecData
from continuum import data_from_dir

class StellgapSimTools:
    _plasma_data: PlasmaDat
    _fourier_data: FourierDat
    _alfven_spec_data: AlfvenSpecData
    
    def __init__(self, plasma_data: str = None, fourier_data: str = None, alfven_spec_directory: str = None):
        """
        TODO: info
        """
        if plasma_data != None:
            self.load_plasma_dat(plasma_data)

        if fourier_data != None:
            self.load_fourier_dat(fourier_data)

        if alfven_spec_directory != None:
            self.load_alfven_spec_data(alfven_spec_directory)
    
    def load_plasma_dat(self, plasma_data: str):
        """
        TODO: info
        """
        # create an empty PlasmaDat object
        self._plasma_data = PlasmaDat()

        self._plasma_data.from_file(plasma_data)

    def load_fourier_dat(self, fourier_data: str):
        """
        TODO: info
        """
        # create an empty FourierDat object
        self._fourier_data = FourierDat()

        self._fourier_data.from_file(fourier_data)
    
    def load_alfven_spec_data(self, directory: str):
        self._alfven_spec_data = data_from_dir(directory)

    def plot_continuum(self, show_legend = False, normalized=False, *args, **kwargs) -> go.Figure:
        # TODO: check if alfven spec data is empty
        omega_A: float = 1

        if normalized:
            # TODO
            print("WIP: Need to add normalization")

        modes = self._alfven_spec_data.get_modes()

        fig = go.Figure()
        for md in modes:
            fig.add_trace(go.Scatter(
                x=md.get_flux_surfaces(),
                y=md.get_frequencies(),
                mode='markers',
                name=f'm={md.get_poloidal_mode()}, n={md.get_toroidal_mode()}',
                marker=dict(size=3),
                line=dict(width=0.5)  # Equivalent to lw in matplotlib
            ))

        fig.update_layout(
        autosize=True,
        title=r'$\text{Continuum: }$',
        xaxis_title=r'$\text{Normalized flux }s$',
        yaxis_title=r'$\text{Frequency }\omega\text{ [kHz]}$',
        xaxis=dict(range=[np.min([np.min(md.get_flux_surfaces()) for md in modes]), np.max([np.max(md.get_flux_surfaces()) for md in modes])]),
        yaxis=dict(range=[0, 600]),
        legend=dict(
            title=r'$\text{Mode: }$',
            yanchor="top",
            y=1.4,
            xanchor="center",
            x=0.5,
            orientation="h"
        ),
        showlegend=show_legend
    )
        return fig


