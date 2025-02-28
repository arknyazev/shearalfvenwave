# simsopt.py
import numpy as np

# Try importing SIMSOPT-related functionality
try:
    from simsopt.field.boozermagneticfield import ShearAlfvenHarmonic, ShearAlfvenWavesSuperposition
    from simsopt.field.boozermagneticfield import BoozerMagneticField
    SIMSOPT_AVAILABLE = True
except ImportError:
    SIMSOPT_AVAILABLE = False

from .continuum import AE3DEigenvector

def saw_from_ae3d(eigenvector: AE3DEigenvector, B0_Tesla : BoozerMagneticField, max_dB_normal_by_B0: float = 1e-3, minor_radius_meters = 1.7, phase = 0.0):
    """
    Converts AE3DEigenvector harmonics into ShearAlfvenHarmonics submerged in the given BoozerMagneticField.

    Args:
        eigenvector (AE3DEigenvector): The eigenvector object containing harmonics from the AE3D simulation.
        B0 (BoozerMagneticField): The background magnetic field (computed separately), in Tesla
        max_dB_normal_by_B0 (float): Desired ration of maximum normal B from SAW mode over B0 field
        minor_radius_meters (float): Stellarator's minor radius, in meters. User can get this from VMEC wout equilibrium

    Returns:
        ShearAlfvenWavesSuperposition: A superposition of ShearAlfvenHarmonics.
    """
    harmonic_list = []
    m_list = []
    n_list = []
    s_list = []
    omega = np.sqrt(eigenvector.eigenvalue)

    if eigenvector.eigenvalue <= 0:
        raise ValueError("The eigenvalue must be positive to compute omega.")

    for harmonic in eigenvector.harmonics:
        sbump = eigenvector.s_coords
        bump = harmonic.amplitudes

        sah = ShearAlfvenHarmonic(
            Phihat_value_or_tuple=(sbump, bump),
            Phim=harmonic.m,
            Phin=harmonic.n,
            omega=omega,
            phase=phase,
            B0=B0_Tesla
        )
        m_list.append(harmonic.m)
        n_list.append(harmonic.n)
        s_list += list(sbump)
        harmonic_list.append(sah)
    #start with arbitrary magnitude SAW; then, rescale it:
    unscaled_SAW = ShearAlfvenWavesSuperposition(harmonic_list)
    #Make radial grid that captures all unique radial values for all harmonic:
    s_unique = list(set(s_list))
    s_unique.sort()
    #Make angle grids that resolve maxima of highest harmonics
    thetas = np.linspace(0, 2 * np.pi, 5*np.max(np.abs(m_list)))
    zetas  = np.linspace(0, 2 * np.pi, 5*np.max(np.abs(n_list)))
    print(f'{np.max(m_list)=}')
    print(f'{np.max(n_list)=}')
    # Create 3D mesh grids:
    thetas2d, zetas2d, s2d = np.meshgrid(thetas, zetas, s_unique, indexing='ij')
    points = np.zeros((len(thetas2d.flatten()), 4)) #s theta zeta time
    points[:, 0] = s2d.flatten()  # s values
    points[:, 1] = thetas2d.flatten()  # theta values
    points[:, 2] = zetas2d.flatten()  # zeta values
    unscaled_SAW.set_points(points)
    G = unscaled_SAW.B0.G()
    iota = unscaled_SAW.B0.iota()
    I = unscaled_SAW.B0.I()
    Bpsi_default = (1/((iota*I+G)*minor_radius_meters)
                *(G*unscaled_SAW.dalphadtheta() - I*unscaled_SAW.dalphadzeta()))
    max_Bpsi_value = np.max(np.abs(Bpsi_default))
    max_index = np.argmax(np.abs(Bpsi_default))
    max_s, max_theta, max_zeta = points[max_index, 0], points[max_index, 1], points[max_index, 2]
    
    print(f'Max |Bpsi_default|: {max_Bpsi_value}')
    print(f'At s={max_s}, theta={max_theta}, zeta={max_zeta}')

    Phihat_scale_factor = max_dB_normal_by_B0/np.max(np.abs(Bpsi_default))
    print(f'{Phihat_scale_factor=}')
    #Having determine the scale factor, initialize harmonics with corrected amplitudes:
    harmonic_list = []
    for harmonic in eigenvector.harmonics:
        sbump = eigenvector.s_coords
        bump = harmonic.amplitudes
        sah = ShearAlfvenHarmonic(
            Phihat_value_or_tuple = (sbump, bump*Phihat_scale_factor),
            Phim = harmonic.m,
            Phin = harmonic.n,
            omega = omega,
            phase = phase,
            B0 = B0_Tesla
        )
        harmonic_list.append(sah)
    return ShearAlfvenWavesSuperposition(harmonic_list)

def rescale_ShearAlfvenHarmonic(max_dB_normal_by_B0: float, unscaled_SAH : ShearAlfvenHarmonic, minor_radius_meters=1.7, phase=0.0, verbose=False):
    '''
    Rescales the amplituded of a shear_alfven_Harmonic to desired value of
    maximum normal component of dB:
    '''
    thetas = np.linspace(0, 2 * np.pi, 5*unscaled_SAH.Phim)
    zetas  = np.linspace(0, 2 * np.pi, 5*unscaled_SAH.Phin)
    if unscaled_SAH.Phihat.get_s_basis() == [0.0, 1.0]:
        if verbose:
            print('Unscale_SAW is a constant Phihat harmonic')
        const_SAH = True
        s_unique = np.linspace(0.0,1.0,1000)
    else:
        if verbose:
            print('Unscale_SAW is a varying Phihat harmonic')
        const_SAH = False
        s_unique = unscaled_SAH.Phihat.get_s_basis()
        if verbose:
            print(f'{s_unique=}')
    thetas2d, zetas2d, s2d = np.meshgrid(thetas, zetas, s_unique, indexing='ij')
    points = np.zeros((len(thetas2d.flatten()), 4)) #s theta zeta time
    points[:, 0] = s2d.flatten()  # s values
    points[:, 1] = thetas2d.flatten()  # theta values
    points[:, 2] = zetas2d.flatten()  # zeta values
    unscaled_SAH.set_points(points)
    G = unscaled_SAH.B0.G()
    iota = unscaled_SAH.B0.iota()
    I = unscaled_SAH.B0.I()
    Bpsi_default = (1/((iota*I+G)*minor_radius_meters)
                *(G*unscaled_SAH.dalphadtheta() - I*unscaled_SAH.dalphadzeta()))
    Phihat_scale_factor = max_dB_normal_by_B0/np.max(np.abs(Bpsi_default))
    if verbose:
        print(f'{Phihat_scale_factor=}')
    if const_SAH:
        if verbose:
            print(f'{unscaled_SAH.Phihat(0.5)=}')
        Phihat_value_or_tuple = unscaled_SAH.Phihat(0.5)*Phihat_scale_factor
    else:
        Phihat_value_or_tuple = (list(s_unique),
            list(unscaled_SAH.Phihat(s)*Phihat_scale_factor for s in s_unique))
    sah = ShearAlfvenHarmonic(
            Phihat_value_or_tuple = Phihat_value_or_tuple,
            Phim = unscaled_SAH.Phim,
            Phin = unscaled_SAH.Phin,
            omega = unscaled_SAH.omega,
            phase = phase,
            B0 = unscaled_SAH.B0
        )
    return sah

def rescaled_ShearAlfvenHarmonic_get_volume_average_Bpsi(max_dB_normal_by_B0: float, unscaled_SAH : ShearAlfvenHarmonic, minor_radius_meters=1.7, phase=0.0, verbose=False):
    '''
    Rescales the amplituded of a shear_alfven_Harmonic to desired value of
    maximum normal component of dB.
    Then, calculates the volume average of Bpsi
    '''
    thetas = np.linspace(0, 2 * np.pi, 5*unscaled_SAH.Phim)
    zetas  = np.linspace(0, 2 * np.pi, 5*unscaled_SAH.Phin)
    if unscaled_SAH.Phihat.get_s_basis() == [0.0, 1.0]:
        if verbose:
            print('Unscale_SAW is a constant Phihat harmonic')
        const_SAH = True
        s_unique = np.linspace(0.0,1.0,1000)
    else:
        if verbose:
            print('Unscale_SAW is a varying Phihat harmonic')
        const_SAH = False
        s_unique = unscaled_SAH.Phihat.get_s_basis()
        if verbose:
            print(f'{s_unique=}')
    thetas2d, zetas2d, s2d = np.meshgrid(thetas, zetas, s_unique, indexing='ij')
    points = np.zeros((len(thetas2d.flatten()), 4)) #s theta zeta time
    points[:, 0] = s2d.flatten()  # s values
    points[:, 1] = thetas2d.flatten()  # theta values
    points[:, 2] = zetas2d.flatten()  # zeta values
    unscaled_SAH.set_points(points)
    G = unscaled_SAH.B0.G()
    iota = unscaled_SAH.B0.iota()
    I = unscaled_SAH.B0.I()
    Bpsi_default = (1/((iota*I+G)*minor_radius_meters)
                *(G*unscaled_SAH.dalphadtheta() - I*unscaled_SAH.dalphadzeta()))
    Phihat_scale_factor = max_dB_normal_by_B0/np.max(np.abs(Bpsi_default))
    if verbose:
        print(f'{Phihat_scale_factor=}')
    if const_SAH:
        if verbose:
            print(f'{unscaled_SAH.Phihat(0.5)=}')
        Phihat_value_or_tuple = unscaled_SAH.Phihat(0.5)*Phihat_scale_factor
    else:
        Phihat_value_or_tuple = (list(s_unique),
            list(unscaled_SAH.Phihat(s)*Phihat_scale_factor for s in s_unique))
    sah = ShearAlfvenHarmonic(
            Phihat_value_or_tuple = Phihat_value_or_tuple,
            Phim = unscaled_SAH.Phim,
            Phin = unscaled_SAH.Phin,
            omega = unscaled_SAH.omega,
            phase = phase,
            B0 = unscaled_SAH.B0
        )
    sah.set_points(points)
    G = sah.B0.G()
    iota = sah.B0.iota()
    I = sah.B0.I()
    modB = sah.B0.modB()
    Bpsi_saw = (1/((iota*I+G)*minor_radius_meters)
                *(G*sah.dalphadtheta() - I*sah.dalphadzeta()))
    #volume average computation:
    sqrtg = (iota*I+G)/modB/modB
    volume_average_B_psi = np.sum(sqrtg*np.sqrt(Bpsi_saw*Bpsi_saw)) / np.sum(sqrtg)
    return volume_average_B_psi

def saw_from_ae3d_get_volume_average_Bpsi(eigenvector: AE3DEigenvector, B0_Tesla : BoozerMagneticField, max_dB_normal_by_B0: float = 1e-3, minor_radius_meters = 1.7, phase = 0.0):
    """
    Converts AE3DEigenvector harmonics into ShearAlfvenHarmonics submerged in the given BoozerMagneticField.
    Then, calculates volume_average_Bpsi

    Args:
        eigenvector (AE3DEigenvector): The eigenvector object containing harmonics from the AE3D simulation.
        B0 (BoozerMagneticField): The background magnetic field (computed separately), in Tesla
        max_dB_normal_by_B0 (float): Desired ration of maximum normal B from SAW mode over B0 field
        minor_radius_meters (float): Stellarator's minor radius, in meters. User can get this from VMEC wout equilibrium

    Returns:
        ShearAlfvenWavesSuperposition: A superposition of ShearAlfvenHarmonics.
    """
    harmonic_list = []
    m_list = []
    n_list = []
    s_list = []
    omega = np.sqrt(eigenvector.eigenvalue)

    if eigenvector.eigenvalue <= 0:
        raise ValueError("The eigenvalue must be positive to compute omega.")

    for harmonic in eigenvector.harmonics:
        sbump = eigenvector.s_coords
        bump = harmonic.amplitudes

        sah = ShearAlfvenHarmonic(
            Phihat_value_or_tuple=(sbump, bump),
            Phim=harmonic.m,
            Phin=harmonic.n,
            omega=omega,
            phase=phase,
            B0=B0_Tesla
        )
        m_list.append(harmonic.m)
        n_list.append(harmonic.n)
        s_list += list(sbump)
        harmonic_list.append(sah)
    #start with arbitrary magnitude SAW; then, rescale it:
    unscaled_SAW = ShearAlfvenWavesSuperposition(harmonic_list)
    #Make radial grid that captures all unique radial values for all harmonic:
    s_unique = list(set(s_list))
    s_unique.sort()
    #Make angle grids that resolve maxima of highest harmonics
    thetas = np.linspace(0, 2 * np.pi, 5*np.max(m_list))
    zetas  = np.linspace(0, 2 * np.pi, 5*np.max(n_list))
    print(f'{np.max(m_list)=}')
    print(f'{np.max(n_list)=}')
    # Create 3D mesh grids:
    thetas2d, zetas2d, s2d = np.meshgrid(thetas, zetas, s_unique, indexing='ij')
    points = np.zeros((len(thetas2d.flatten()), 4)) #s theta zeta time
    points[:, 0] = s2d.flatten()  # s values
    points[:, 1] = thetas2d.flatten()  # theta values
    points[:, 2] = zetas2d.flatten()  # zeta values
    unscaled_SAW.set_points(points)
    G = unscaled_SAW.B0.G()
    iota = unscaled_SAW.B0.iota()
    I = unscaled_SAW.B0.I()
    Bpsi_default = (1/((iota*I+G)*minor_radius_meters)
                *(G*unscaled_SAW.dalphadtheta() - I*unscaled_SAW.dalphadzeta()))
    Phihat_scale_factor = max_dB_normal_by_B0/np.max(np.abs(Bpsi_default))
    print(f'{Phihat_scale_factor=}')
    #Having determine the scale factor, initialize harmonics with corrected amplitudes:
    harmonic_list = []
    for harmonic in eigenvector.harmonics:
        sbump = eigenvector.s_coords
        bump = harmonic.amplitudes
        sah = ShearAlfvenHarmonic(
            Phihat_value_or_tuple = (sbump, bump*Phihat_scale_factor),
            Phim = harmonic.m,
            Phin = harmonic.n,
            omega = omega,
            phase = phase,
            B0 = B0_Tesla
        )
        harmonic_list.append(sah)
    SAW = ShearAlfvenWavesSuperposition(harmonic_list)
    SAW.set_points(points)
    G = SAW.B0.G()
    iota = SAW.B0.iota()
    I = SAW.B0.I()
    modB = SAW.B0.modB()
    Bpsi_saw = (1/((iota*I+G)*minor_radius_meters)
                *(G*SAW.dalphadtheta() - I*SAW.dalphadzeta()))
    #volume average computation:
    sqrtg = (iota*I+G)/modB/modB
    volume_average_B_psi = np.sum(sqrtg*np.sqrt(Bpsi_saw*Bpsi_saw)) / np.sum(sqrtg)
    return volume_average_B_psi
