from .continuum import FourierDat, PlasmaDat, TaeDataBoozer, AeMetricData #inputs from Stellgap and AE3D
from .continuum import Mode, AlfvenSpecData, ModesOutput, DataPost #outputs from Stellgap and AE3D
from .continuum import IonProfile, ProfilesDat, EgnvaluesDat #outputs from Stellgap and AE3D
from .continuum import FieldBendingMatrix, InertiaMatrix, Omega2Dat, EigModeASCI #outputs from AE3D
from .continuum import AE3DEigenvector, Harmonic #outputs from AE3D
from .continuum import FAR3DEigenproblem
from .continuum import plot_continuum, plot_condition_numbers, data_from_dir, plot_ae3d_eigenmode
from .continuum import continuum_from_ae3d, plot_continua, numpy_eig_to_EigModeASCI
from .continuum import Mode
# Try to import SIMSOPT functions
try:
    from .simsopt import saw_from_ae3d, rescale_ShearAlfvenHarmonic
    from .simsopt import saw_from_ae3d_get_volume_average_Bpsi #temporary function, rewrite
    from .simsopt import rescaled_ShearAlfvenHarmonic_get_volume_average_Bpsi #temporary function, rewrite
except ImportError:
    print("SIMSOPT-related functionality is not available. Please install SIMSOPT to use it.")
