import stellgap as sg
import numpy as np
import matplotlib.pyplot as plt

example="CTOK"
solver = sg.AE3DEigensolver(example, scale_factor=1)

eigenvalues, eigenvectors = solver.solve_eigenproblem(method='dense')

print("\nComparing eigenvalues with AE3D results:")
comparison = solver.compare_with_ae3d(top_n=10)
if comparison is not None:
    print(f"{'Index':<6}{'AE3D':<15}{'Computed':<15}{'Diff (%)':<10}")
    print("-" * 46)
    for i, (ae3d, comp, diff) in enumerate(zip(
            comparison['ae3d_eigenvalues'],
            comparison['computed_eigenvalues'],
            comparison['relative_difference_percent'])):
        print(f"{i:<6}{ae3d:<15.6e}{comp:<15.6e}{diff:<10.3f}")
else:
    print("No AE3D eigenvalues available for comparison.")

solver.plot_eigenvalues(compare_with_ae3d=True)

print("\nLoading EigModeASCI data...")
try:
    ema = sg.EigModeASCI(example)
    print(f"Number of eigenmodes in EigModeASCI: {ema.num_eigenmodes}")
    print(f"Number of Fourier modes: {ema.num_fourier_modes}")
    print(f"Number of radial points: {ema.num_radial_points}")
    
    print("\nComparison between EigModeASCI eigenvalues"
          " and computed eigenvalues:")
    print(f"{'Index':<6}{'EigModeASCI':<15}{'Computed':<15}{'Diff (%)':<10}")
    print("-" * 46)
    
    # Sort both sets of eigenvalues by real part for comparison (largest first)
    ema_eigenvalues = np.sort(ema.egn_values)[::-1]
    computed_eigenvalues = np.sort(np.real(eigenvalues))[::-1]
    
    assert len(ema_eigenvalues) == len(computed_eigenvalues), "Wrong eigenvalue count"
    for i in range(len(ema_eigenvalues)):
        ema_val = ema_eigenvalues[i]
        comp_val = computed_eigenvalues[i]
        diff_percent = np.abs(ema_val - comp_val) / np.abs(ema_val) * 100 if abs(ema_val) > 1e-10 else 0
        print(f"{i:<6}{ema_val:<15.6e}{comp_val:<15.6e}{diff_percent:<10.3f}")
        
except FileNotFoundError:
    print("EigModeASCI data not found. You may need to create it first.")

print("\nSaving results to eig_mode_asci.dat format...")
import os

output_dir = os.path.join(example, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

solver.save_to_eig_mode_asci(output_dir)
print(f"Results saved to {os.path.join(output_dir, 'egn_mode_asci.dat')}")

print("\nLoading the saved results to verify:")
try:
    saved_ema = sg.EigModeASCI(output_dir)
    print(f"Successfully loaded {saved_ema.num_eigenmodes} eigenmodes from saved file")
    
    # Verify the eigenvalues match our computed ones (preserving signs)
    print("\nVerifying saved eigenvalues match computed eigenvalues:")
    print(f"{'Index':<6}{'Saved':<15}{'Original':<15}{'Match?':<10}")
    print("-" * 46)
    
    # Sort the original eigenvalues by real part
    real_eigenvalues = np.real(eigenvalues)
    sorted_indices = np.argsort(-real_eigenvalues)  # Negative sign for descending order
    sorted_original_eigenvalues = real_eigenvalues[sorted_indices]
    
    for i in range(saved_ema.num_eigenmodes):
        saved_val = saved_ema.egn_values[i]
        orig_val = sorted_original_eigenvalues[i]
        is_match = np.isclose(saved_val, orig_val, rtol=1e-5)
        print(f"{i:<6}{saved_val:<15.6e}{orig_val:<15.6e}{'Yes' if is_match else 'No':<10}")
        
except FileNotFoundError:
    print("Could not load the saved file. Check the output directory.")
