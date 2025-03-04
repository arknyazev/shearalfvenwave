import os
import numpy as np
from typing import List
from dataclasses import dataclass
import plotly.graph_objects as go

@dataclass
class Mode:
    n: int
    m: int
    s: np.ndarray
    freq: np.ndarray

def get_modes(alfven_spec_filenames: List[str]) -> List[Mode]:
    assert len(alfven_spec_filenames) > 0, "No alfven_spec_filenames provided"
    data = np.vstack([np.loadtxt(filename,
        dtype=[('s', float),  #normalized flux
               ('ar', float), #real part of eigenvalue
               ('ai', float), #imaginary part of eigenvalue (always zero for ideal MHD)
               ('beta', float), #what is this?
               ('m', int),    #poloidal mode number
               ('n', int)]) #toroidal mode number
        for filename in alfven_spec_filenames])
    data = data[data['beta'] != 0]
    modes = [
        Mode(
            n=n, 
            m=m, 
            s=(filtered_data := np.sort(data[(data['n'] == n) & (data['m'] == m)], order='s'))['s'], 
            freq=np.sqrt(np.abs(filtered_data['ar'] / filtered_data['beta'])))
        for n, m in {(a['n'], a['m']) for a in data}
        ]
    return modes

def plot_continuum(modes: List[Mode]) -> None:
    fig = go.Figure()
    for md in modes:
        fig.add_trace(go.Scatter(
            x=md.s,
            y=md.freq,
            mode='lines+markers',
            name=f'm={md.m}, n={md.n}',
            marker=dict(size=3),
            line=dict(width=0.5)  # Equivalent to lw in matplotlib
        ))

    fig.update_layout(
    autosize=True,
    title=r'$\text{Continuum: }$',
    xaxis_title=r'$\text{Normalized flux }s$',
    yaxis_title=r'$\text{Frequency }\omega\text{ [kHz]}$',
    xaxis=dict(range=[np.min([np.min(md.s) for md in modes]), np.max([np.max(md.s) for md in modes])]),
    yaxis=dict(range=[0, 600]),
    legend=dict(
        title=r'$\text{Mode: }$',
        yanchor="top",
        y=1.4,
        xanchor="center",
        x=0.5,
        orientation="h"
    )
)
    return fig

if __name__ == '__main__':
    '''Run this script to plot the continuum of Alfven modes in the current directory.'''
    modes = get_modes([fname for fname in os.listdir(os.getcwd()) if fname.startswith('alfven_spec')])
    assert len(modes) > 0, "No Alfven mode data found in the current directory."
    modes.sort(key=lambda md: md.n)
    fig = plot_continuum(list(filter(lambda md: np.any(md.freq < 600), modes)))
    fig.show()