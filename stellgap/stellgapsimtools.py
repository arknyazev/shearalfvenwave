import os
import numpy as np

from continuum import PlasmaDat
from continuum import FourierDat

class StellgapSimTools:
    _plasma_data: PlasmaDat
    _fourier_data: FourierDat

    def __init__(self):
        """
        Empty constructor
        """

        return 0
    
    def __init__(self, plasma_data: str, fourier_data: str):
        """
        TODO: info
        """
        self.load_plasma_dat(plasma_data)
        self.load_fourier_dat(fourier_data)

        return 0
    
    def load_plasma_dat(self, plasma_data: str):
        """
        TODO: info
        """
        # create an empty PlasmaDat object
        self._plasma_data = PlasmaDat()

        self._plasma_data.from_file(plasma_data)

        return 0

    def load_fourier_dat(self, fourier_data: str):
        """
        TODO: info
        """
        # create an empty PlasmaDat object
        self._fourier_data = PlasmaDat()

        self._fourier_data.from_file(fourier_data)

        return 0
    
    # def load_alfven_spec_data()

    # def plot_continuum(normalized=False,*args,**kwargs):


