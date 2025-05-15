import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg
import argparse
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import time
import logging
from .continuum import (
    FieldBendingMatrix,
    InertiaMatrix,
    Harmonic,
    AE3DEigenvector
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AE3D-Eigensolver")

class AE3DEigensolver:
    """
    A class that solves the AE3D eigenproblem using matrix outputs.
    
    This class loads matrices from a_matrix.dat and b_matrix.dat files,
    solves the generalized eigenvalue problem Ax = λBx, and 
    processes the results to create AE3DEigenvector objects.
    """
    
    def __init__(self,
        sim_dir: str,
        symmetric: bool = True,
        scale_factor: float = 1.0) -> None:
        """
        Initialize the eigensolver with the directory 
        containing the matrix files.
        
        Args:
            sim_dir (str): Directory containing AE3D output files
            symmetric (bool): Whether to force matrices to be symmetric
            scale_factor (float): Factor to scale the B matrix 
                                  if needed for numerical stability
        """
        self.sim_dir = sim_dir
        self.symmetric = symmetric
        self.scale_factor = scale_factor
        self.A = None  # Field bending matrix
        self.B = None  # Inertia matrix
        self.A_dense = None  # Dense version of A
        self.B_dense = None  # Dense version of B
        self.eigenvalues = None
        self.eigenvectors = None
        self.modes = None
        self.s_coords = None
        self.num_fourier_modes = None
        self.num_radial_points = None
        
        self.load_matrices()
        
        # Try to load mode information if possible
        self.load_mode_info()
        
        # Load AE3D eigenvalues if available for comparison
        self.ae3d_eigenvalues = self.load_ae3d_eigenvalues()
    
    def load_matrices(self):
        """Load A and B matrices from the simulation directory"""
        t1 = time.time()
        logger.info(f"Loading matrices from {self.sim_dir}")
        
        self.A = FieldBendingMatrix(self.sim_dir).matrix
        self.B = InertiaMatrix(self.sim_dir).matrix
            
        # Force symmetry if requested, done in AE3D solver.
        if self.symmetric:
            self.A = 0.5 * (self.A + self.A.T)
            self.B = 0.5 * (self.B + self.B.T)
        
        # Scale B matrix for numerical stability
        self.B = self.B * self.scale_factor
        
        # Convert to CSR format for sparce solver
        self.A = self.A.tocsr()
        self.B = self.B.tocsr()
        
        # Also create dense matrices for dense solver
        self.A_dense = self.A.toarray()
        self.B_dense = self.B.toarray()
        
        t2 = time.time()
        logger.info(f"Matrices loaded in {t2 - t1:.2f} seconds")
        
        # Note percentage of non-zeros for each matrix
        a_size = self.A.shape[0]
        a_nnz_percent = self.A.nnz / (a_size * a_size) * 100
        b_size = self.B.shape[0]
        b_nnz_percent = self.B.nnz / (b_size * b_size) * 100

        logger.info(f"A: {self.A.shape}, {self.A.nnz} non-zeros "
                    f"({a_nnz_percent:.2f}%)")
        logger.info(f"B: {self.B.shape}, {self.B.nnz} non-zeros "
                    f"({b_nnz_percent:.2f}%)")
    
    def load_mode_info(self):
        """Attempt to load mode information from relevant files"""
        # Try to load egn_mode_asci.dat if available
        egn_mode_file = os.path.join(self.sim_dir, 'egn_mode_asci.dat')
        
        if os.path.isfile(egn_mode_file):
            logger.info(f"Loading mode information from {egn_mode_file}")
            try:
                data = np.loadtxt(egn_mode_file)
                it = iter(data)
                
                num_eigenmodes = int(next(it))
                self.num_fourier_modes = int(next(it))
                self.num_radial_points = int(next(it))
                
                # Load mode numbers
                self.modes = np.array(
                    [(next(it), next(it)) 
                        for _ in range(self.num_fourier_modes)],
                    dtype=[('m', 'int32'), ('n', 'int32')]
                    )
                
                # Skip eigenvalues
                for _ in range(num_eigenmodes):
                    next(it)
                
                # Load s coordinates
                self.s_coords = np.array(
                    [next(it) for _ in range(self.num_radial_points)]
                )
                
                logger.info(f"Successfully loaded mode information: "
                           f"{self.num_fourier_modes} Fourier modes, "
                           f"{self.num_radial_points} radial points")
            except Exception as e:
                logger.warning(f"Failed to load mode information: {str(e)}")
        else:
            logger.warning("Could not find egn_mode_asci.dat "
                           "for mode information")
            
            # TODO: try to use other files that contain mode information
            #       as potential additional sources of mode data
            pass
    
    def load_ae3d_eigenvalues(self) -> Optional[np.ndarray]:
        """Load eigenvalues from egn_values.dat if available for comparison"""
        egn_file = os.path.join(self.sim_dir, 'egn_values.dat')
        
        if os.path.isfile(egn_file):
            logger.info(f"Loading AE3D eigenvalues from {egn_file}")
            try:
                data = np.loadtxt(egn_file, dtype=[
                    ('eigenvalue', 'f8'),
                    ('electrostatic_energy', 'f8'),
                    ('electromagnetic_energy', 'f8')
                ])
                
                # AE3D eigenvalues in the file: 
                # (positive values are square roots of SAW eigenvalues)
                eigenvalues = data['eigenvalue'].copy()
                positive_mask = eigenvalues > 0
                eigenvalues[positive_mask] = eigenvalues[positive_mask]**2
                
                logger.info(f"Loaded {len(eigenvalues)} eigenvalues from AE3D")
                return eigenvalues
            except Exception as e:
                logger.warning(f"Failed to load AE3D eigenvalues: {str(e)}")
        else:
            logger.warning("Could not find egn_values.dat "
                           "for AE3D eigenvalues")
        return None
    
    def solve_eigenproblem_sparse(
        self,
        k: int = 10, 
        sigma: Optional[float] = None, 
        which: str = 'SM', 
        tol: float = 1e-4, 
        return_eigenvectors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Solve the generalized eigenvalue problem Ax = λBx using sparse methods
        for a subset of eigenvalues.
        
        Args:
            k (int): Number of eigenvalues/vectors to compute
            sigma (float, optional): Target value for shift-invert mode
            which (str): Which eigenvalues to find:
                         'LM' = largest magnitude
                         'SM' = smallest magnitude
                         'LR' = largest real part
                         'SR' = smallest real part
            tol (float): Relative accuracy for eigenvalues
            return_eigenvectors (bool): Whether to compute eigenvectors
            
        Returns:
            tuple: (eigenvalues, eigenvectors) if return_eigenvectors=True
                   eigenvalues only otherwise
        """
        logger.info(f"Solving sparse eigenproblem for {k} eigenvalues")
        t1 = time.time()
        
        if sigma is not None:
            logger.info(f"Using shift-invert mode with sigma={sigma}")
            
        try:
            if return_eigenvectors:
                eigenvalues, eigenvectors = spla.eigsh(
                    self.A,
                    k=k,
                    M=self.B, 
                    sigma=sigma,
                    which=which, 
                    tol=tol,
                    return_eigenvectors=True)
                
                # Rescale eigenvalues to undo scaling of B
                eigenvalues = eigenvalues * self.scale_factor
                
                self.eigenvalues = eigenvalues
                self.eigenvectors = eigenvectors
                
                t2 = time.time()
                logger.info("Sparse eigensolve completed "
                            f"in {t2 - t1:.2f} seconds")
                
                return eigenvalues, eigenvectors
            else:
                eigenvalues = spla.eigsh(self.A, k=k, M=self.B, 
                                        sigma=sigma, which=which, 
                                        tol=tol, return_eigenvectors=False)
                
                # Rescale eigenvalues to undo scaling of B
                eigenvalues = eigenvalues * self.scale_factor
                self.eigenvalues = eigenvalues
                
                t2 = time.time()
                logger.info("Sparse eigensolve completed in "
                            f"{t2 - t1:.2f} seconds")
                
                return eigenvalues
        except Exception as e:
            logger.error(f"Sparse eigensolve failed: {str(e)}")
            raise
    
    def solve_eigenproblem_dense(
        self,
        return_eigenvectors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Solve the generalized eigenvalue problem Ax = λBx using dense methods
        to find all eigenvalues, equivalent to DGGEV.
        
        Args:
            return_eigenvectors (bool): Whether to compute eigenvectors
            
        Returns:
            tuple: (eigenvalues, eigenvectors) if return_eigenvectors=True
                   eigenvalues only otherwise
        """
        logger.info(f"Solving dense eigenproblem for all eigenvalues")
        t1 = time.time()
        
        try:
            if return_eigenvectors:
                eigenvalues, eigenvectors_right = scipy.linalg.eig(
                    self.A_dense,
                    self.B_dense,
                    right=True
                )
                
                # Rescale eigenvalues to undo scaling of B
                eigenvalues = eigenvalues * self.scale_factor
                
                self.eigenvalues = eigenvalues
                self.eigenvectors = eigenvectors_right
                
                t2 = time.time()
                logger.info("Dense eigensolve completed in "
                            f"{t2 - t1:.2f} seconds"
                )
                
                return eigenvalues, eigenvectors_right
            else:
                eigenvalues = scipy.linalg.eigvals(self.A_dense, self.B_dense)
                
                # Rescale eigenvalues to undo scaling of B
                eigenvalues = eigenvalues * self.scale_factor
                
                self.eigenvalues = eigenvalues
                
                t2 = time.time()
                logger.info("Dense eigensolve completed"
                            f"in {t2 - t1:.2f} seconds")
                
                return eigenvalues
        except Exception as e:
            logger.error(f"Dense eigensolve failed: {str(e)}")
            raise
    
    def solve_eigenproblem(
        self,
        method: str = 'sparse',
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Solve the generalized eigenvalue problem using the specified method.
        
        Args:
            method (str): Method to use ('sparse' or 'dense')
            **kwargs: Additional arguments passed to the specific solver method
            
        Returns:
            Result from the chosen solver method
        """
        if method.lower() == 'sparse':
            return self.solve_eigenproblem_sparse(**kwargs)
        elif method.lower() == 'dense':
            return self.solve_eigenproblem_dense(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sparse' or 'dense'.")
    
    def compare_with_ae3d(
        self, 
        top_n: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Compare computed eigenvalues with AE3D eigenvalues

        Args:
            top_n (int): Number of top eigenvalues to compare

        Returns:
            Dict[str, np.ndarray]: Dictionary containing comparison data
            Contains:
                - 'ae3d_eigenvalues'
                - 'computed_eigenvalues'
                - 'relative_difference_percent'
        """
        if self.eigenvalues is None:
            raise ValueError("No eigenvalues computed. "
                             "Run solve_eigenproblem first.")

        if self.ae3d_eigenvalues is None:
            logger.warning("No AE3D eigenvalues available for comparison.")
            return None

        # Sort both sets of eigenvalues
        computed_vals = np.sort(np.abs(self.eigenvalues.real))[::-1][:top_n]
        ae3d_vals = np.sort(np.abs(self.ae3d_eigenvalues))[::-1][:top_n]

        # Calculate relative differences
        min_length = min(len(computed_vals), len(ae3d_vals))
        computed_vals = computed_vals[:min_length]
        ae3d_vals = ae3d_vals[:min_length]

        rel_diff = np.abs(computed_vals - ae3d_vals) / np.abs(ae3d_vals) * 100

        comparison = {
            'ae3d_eigenvalues': ae3d_vals,
            'computed_eigenvalues': computed_vals,
            'relative_difference_percent': rel_diff
        }

        return comparison
    
    def reshape_eigenvector(
        self,
        vector_index: int = 0
    ) -> np.ndarray:
        """
        Reshape the eigenvector to the format expected by AE3DEigenvector
        
        Args:
            vector_index (int): Index of the eigenvector to reshape
            
        Returns:
            np.ndarray: Reshaped eigenvector
        """
        if self.eigenvectors is None:
            raise ValueError("Eigenvectors not computed. "
                             "Run solve_eigenproblem first.")
        
        if self.num_fourier_modes is None or self.num_radial_points is None:
            raise ValueError("Mode information not available."
                             "Cannot reshape eigenvector.")
        
        vector = self.eigenvectors[:, vector_index]
        
        reshaped = np.zeros(
            (self.num_radial_points, self.num_fourier_modes),
            dtype=complex
        )
        
        for i in range(self.num_radial_points):
            start_idx = i * self.num_fourier_modes
            end_idx = start_idx + self.num_fourier_modes
            if end_idx <= len(vector):
                reshaped[i, :] = vector[start_idx:end_idx]
        
        return reshaped
    
    def create_ae3d_eigenvector(
        self, 
        vector_index: int = 0
    ) -> Union[object, Dict]:
        """
        Create an AE3DEigenvector object from the computed eigensolution
        
        Args:
            vector_index (int): Index of the eigenvector to use
            
        Returns:
            AE3DEigenvector
        """
        if not self.modes is not None or self.s_coords is None:
            raise ValueError("Mode information not available."
                             "Cannot create AE3DEigenvector.")
        
        eigenvalue = self.eigenvalues[vector_index]
        reshaped_vector = self.reshape_eigenvector(vector_index)
        
        assert np.isclose(np.imag(eigenvalue), 0),"ERROR: Complex eivencector."
        
        reshaped_vector = np.real(reshaped_vector)
        eigenvalue = np.real(eigenvalue)
        
        # Find the maximum amplitude to normalize
        max_idx = np.unravel_index(
            np.argmax(
                np.abs(reshaped_vector)
                ), 
            reshaped_vector.shape
            )
        normalization = reshaped_vector[max_idx]
        normalized_vector = reshaped_vector / normalization
        
        # Sort modes by energy
        mode_energies = np.sum(np.abs(normalized_vector)**2, axis=0)
        sort_indices = np.argsort(-mode_energies)
        sorted_modes = self.modes[sort_indices]
        sorted_vector = normalized_vector[:, sort_indices]
        
        harmonics = [
            Harmonic(
                m=sorted_modes['m'][i],
                n=sorted_modes['n'][i],
                amplitudes=sorted_vector[:, i]
                )
            for i in range(len(sorted_modes))
        ]
        
        return AE3DEigenvector(
            eigenvalue=eigenvalue,
            s_coords=self.s_coords,
            harmonics=harmonics
        )
    
    def plot_eigenvalues(self,
        save_path: Optional[str] = None,
        compare_with_ae3d: bool = False
    ) -> None:
        """
        Plot computed eigenvalues
        
        Args:
            save_path (str, optional): Path to save the plot
            compare_with_ae3d (bool): Whether to include AE3D eigenvalues
        """
        if self.eigenvalues is None:
            raise ValueError("Eigenvalues not computed."
                             "Run solve_eigenproblem first.")
        
        plt.figure(figsize=(12, 8))
        
        # Plot real vs imaginary parts if any eigenvalues are complex
        if np.any(np.abs(np.imag(self.eigenvalues)) > 1e-10):
            plt.subplot(121)
            plt.scatter(
                np.real(self.eigenvalues),
                np.imag(self.eigenvalues),
                marker='o',
                label='Computed'
                )
            plt.xlabel('Real part')
            plt.ylabel('Imaginary part')
            plt.grid(True)
            plt.title('Eigenvalue Distribution in Complex Plane')
            
            plt.subplot(122)
            
            # Plot positive real eigenvalues (shear Alfvén frequencies)
            positive_evals = np.real(self.eigenvalues)[
                np.real(self.eigenvalues) > 0
                ]
            indices = np.arange(len(positive_evals))
            plt.plot(indices, positive_evals, 'o-', label='Computed')
            
            if compare_with_ae3d and self.ae3d_eigenvalues is not None:
                # Add AE3D eigenvalues if available
                positive_ae3d = self.ae3d_eigenvalues[
                    self.ae3d_eigenvalues > 0
                    ]
                ae3d_indices = np.arange(len(positive_ae3d))
                plt.plot(ae3d_indices, positive_ae3d, 'x-', label='AE3D')
            
            plt.xlabel('Index')
            plt.ylabel('Eigenvalue')
            plt.grid(True)
            plt.title('Positive Eigenvalues (Shear Alfvén Frequencies)')
            plt.legend()
        else:
            # If all eigenvalues are real, just plot them against their indices
            sorted_evals = np.sort(np.real(self.eigenvalues))
            indices = np.arange(len(sorted_evals))
            
            plt.plot(indices, sorted_evals, 'o-', label='All computed')
            
            positive_evals = sorted_evals[sorted_evals > 0]
            if len(positive_evals) > 0:
                positive_indices = np.arange(len(positive_evals))
                plt.plot(
                    positive_indices,
                    positive_evals,
                    's-',
                    label='Positive computed'
                    )
            
            if compare_with_ae3d and self.ae3d_eigenvalues is not None:
                # Add AE3D eigenvalues if available
                sorted_ae3d = np.sort(self.ae3d_eigenvalues)
                ae3d_indices = np.arange(len(sorted_ae3d))
                plt.plot(ae3d_indices, sorted_ae3d, 'x-', label='AE3D')
            
            plt.xlabel('Index')
            plt.ylabel('Eigenvalue')
            plt.grid(True)
            plt.title('Eigenvalue Spectrum')
            plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Eigenvalue plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_eigenvector(
        self,
        vector_index: int = 0, 
        max_harmonics: int = 5, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the top harmonics of an eigenvector
        
        Args:
            vector_index (int): Index of the eigenvector to plot
            max_harmonics (int): Maximum number of harmonics to display
            save_path (str, optional): Path to save the plot
        """
        if self.eigenvectors is None:
            raise ValueError("Eigenvectors not computed. "
                             "Run solve_eigenproblem first.")
        
        reshaped = self.reshape_eigenvector(vector_index)
        
        if np.any(np.abs(np.imag(reshaped)) > 1e-10):
            logger.warning("Found complex components in eigenvector")
        eigenvalue = self.eigenvalues[vector_index]
        
        energy = np.sum(np.abs(reshaped)**2, axis=0)
        top_indices = np.argsort(-energy)[:max_harmonics]
        
        plt.figure(figsize=(12, 6))
        for i, idx in enumerate(top_indices):
            m, n = self.modes[idx]
            plt.plot(
                self.s_coords,
                np.real(reshaped[:, idx]), 
                label=f'Mode (m={m}, n={n}), Energy={energy[idx]:.2e}'
                )
        
        plt.xlabel('Normalized Radius (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Eigenvector {vector_index}, λ={eigenvalue:.6e}')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Eigenvector plot saved to {save_path}")
        
        plt.show()
    
    def save_eigenvector_to_numpy(
        self, 
        vector_index: int = 0, 
        filename: str = 'eigenvector.npy',
        resolution_step: int = 1
    ) -> None:
        """
        Save eigenvector to numpy file,
        compatible with AE3DEigenvector.load_from_numpy
        
        Args:
            vector_index (int): Index of the eigenvector to save
            filename (str): Output filename
            resolution_step (int): Step size for radial resolution
        """
        if not self.modes is not None or self.s_coords is None:
            raise ValueError("Mode information not available."
                             "Cannot save eigenvector.")
        
        eigenvalue = self.eigenvalues[vector_index]
        reshaped_vector = self.reshape_eigenvector(vector_index)
        
        if np.any(np.abs(np.imag(reshaped_vector)) > 1e-10):
            logger.warning("Found complex components in eigenvector")
        reshaped_vector = np.real(reshaped_vector)
        eigenvalue = np.real(eigenvalue)
        
        # Find the maximum amplitude to normalize
        max_idx = np.unravel_index(
            np.argmax(np.abs(reshaped_vector)), 
            reshaped_vector.shape
            )
        normalization = reshaped_vector[max_idx]
        normalized_vector = reshaped_vector / normalization
        
        # Sort modes by energy
        mode_energies = np.sum(np.abs(normalized_vector)**2, axis=0)
        sort_indices = np.argsort(-mode_energies)
        sorted_modes = self.modes[sort_indices]
        sorted_vector = normalized_vector[:, sort_indices]
        
        # Create format compatible with AE3DEigenvector.load_from_numpy
        harmonics_data = {
            'eigenvalue': eigenvalue,
            's_coords': self.s_coords[0:-1:resolution_step],
            'harmonics': np.array([
                (sorted_modes['m'][i],
                 sorted_modes['n'][i], 
                 sorted_vector[0:-1:resolution_step, i]) 
                for i in range(len(sorted_modes))
                ], 
                dtype=object
                )
        }
        
        np.save(filename, harmonics_data)
        logger.info(f'Harmonics exported to {filename}')
        
    def save_to_eig_mode_asci(
        self,
        output_dir: str = None,
        num_eigenmodes: int = None,
        filename: str = 'egn_mode_asci.dat'
    ) -> None:
        """
        Save eigenvalues and eigenvectors to eig_mode_asci.dat format,
        compatible with EigModeASCI.
        
        This allows for storing the results of eigensolves to be used
        in the same way as AE3D output.
        
        Args:
            output_dir (str): Directory to save the file in. If None, uses self.sim_dir
            num_eigenmodes (int): Number of eigenmodes to save. If None, saves all
            filename (str): Name of the output file
        """
        if self.eigenvalues is None or self.eigenvectors is None:
            raise ValueError("No eigenvalues/eigenvectors computed. "
                             "Run solve_eigenproblem first.")
        
        if not self.modes is not None or self.s_coords is None:
            raise ValueError("Mode information not available. "
                             "Cannot save eigenvectors.")
        
        if output_dir is None:
            output_dir = self.sim_dir
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_path = os.path.join(output_dir, filename)
        
        # Determine how many eigenmodes to save
        if num_eigenmodes is None:
            num_eigenmodes = len(self.eigenvalues)
        else:
            num_eigenmodes = min(num_eigenmodes, len(self.eigenvalues))
        
        # Process eigenvalues:
        # take real part and sort by magnitude (largest first)
        real_eigenvalues = np.real(self.eigenvalues)
        sorted_indices = np.argsort(-np.abs(real_eigenvalues))
        selected_indices = sorted_indices[:num_eigenmodes]
        selected_eigenvalues = real_eigenvalues[selected_indices]
        
        # Process eigenvectors
        processed_eigenvectors = []
        for idx in selected_indices:
            reshaped_vector = self.reshape_eigenvector(idx)
            
            # Use real part of eigenvectors
            if np.any(np.abs(np.imag(reshaped_vector)) > 1e-10):
                logger.warning("Found complex components in eigenvector")
            reshaped_vector = np.real(reshaped_vector)
            
            # Normalize the eigenvector
            max_idx = np.unravel_index(
                np.argmax(np.abs(reshaped_vector)), 
                reshaped_vector.shape
            )
            normalization = reshaped_vector[max_idx]
            normalized_vector = reshaped_vector / normalization
            
            processed_eigenvectors.append(normalized_vector)
        
        # Format data for egn_mode_asci.dat
        with open(file_path, 'w') as f:
            # Write the header: num_eigenmodes, num_fourier_modes, num_radial_points
            f.write(f"{num_eigenmodes}\n")
            f.write(f"{self.num_fourier_modes}\n")
            f.write(f"{self.num_radial_points}\n")
            
            # Write the modes (m, n)
            for m, n in zip(self.modes['m'], self.modes['n']):
                f.write(f"{m}\n{n}\n")
            
            # Write the eigenvalues
            for eigenvalue in selected_eigenvalues:
                f.write(f"{eigenvalue}\n")
            
            # Write the s_coords (radial points)
            for s in self.s_coords:
                f.write(f"{s}\n")
            
            # Write all eigenvector elements
            for i in range(num_eigenmodes):
                vector = processed_eigenvectors[i]
                for j in range(self.num_radial_points):
                    for k in range(self.num_fourier_modes):
                        f.write(f"{vector[j, k]}\n")
        
        logger.info(f"Saved {num_eigenmodes} eigenmodes to "
                    f"{file_path} in EigModeASCI format")


def main():
    """Command line interface for AE3D eigensolver"""
    parser = argparse.ArgumentParser(
        description='Solve AE3D eigenproblems using matrix outputs'
        )

    # Input directory
    parser.add_argument(
        '--sim_dir', 
        type=str, 
        required=True, 
        help='Directory containing AE3D output files'
    )

    # Solution method options
    parser.add_argument(
        '--method', 
        type=str, 
        default='sparse', 
        choices=['sparse', 'dense'],
        help='Method for solving eigenproblem (sparse or dense)'
    )

    # Sparse solver parameters
    parser.add_argument(
        '--k', 
        type=int, 
        default=10, 
        help='Number of eigenvalues to compute (sparse method only)'
    )
    parser.add_argument(
        '--sigma', 
        type=float, 
        default=None, 
        help='Target eigenvalue for shift-invert (sparse method only)'
    )
    parser.add_argument(
        '--which', 
        type=str, 
        default='SM', 
        choices=['LM', 'SM', 'LR', 'SR'],
        help='Which eigenvalues to find (sparse method only)'
    )
    parser.add_argument(
        '--tol', 
        type=float, 
        default=1e-4, 
        help='Tolerance for eigenvalue computation (sparse method only)'
    )

    # General solver options
    parser.add_argument(
        '--scale', 
        type=float, 
        default=1e8, 
        help='Scaling factor for B matrix'
    )

    # Output options
    parser.add_argument(
        '--output', 
        type=str, 
        default=None, 
        help='Output file for eigenvector'
    )
    parser.add_argument(
        '--plot', 
        action='store_true', 
        help='Plot eigenvalues and eigenvector'
    )
    parser.add_argument(
        '--plot_index', 
        type=int, 
        default=0, 
        help='Index of eigenvector to plot'
    )
    parser.add_argument(
        '--save_plots', 
        type=str, 
        default=None, 
        help='Directory to save plots'
    )
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='Compare with AE3D eigenvalues'
    )
    
    args = parser.parse_args()
    
    solver = AE3DEigensolver(args.sim_dir, scale_factor=args.scale)
    
    if args.method == 'sparse':
        eigenvalues, eigenvectors = solver.solve_eigenproblem(
            method='sparse',
            k=args.k,
            sigma=args.sigma,
            which=args.which,
            tol=args.tol
            )
    else:
        eigenvalues, eigenvectors = solver.solve_eigenproblem(method='dense')
    
    logger.info("Computed eigenvalues:")
    num_to_print = min(20, len(eigenvalues))
    for i, val in enumerate(eigenvalues[:num_to_print]):
        logger.info(f"λ_{i} = {val}")
    
    # Compare with AE3D if requested
    if args.compare:
        comparison = solver.compare_with_ae3d()
        if comparison is not None:
            logger.info("Comparison with AE3D eigenvalues:")
            logger.info(str(comparison))
    
    # Save eigenvector if requested
    if args.output:
        solver.save_eigenvector_to_numpy(
            vector_index=args.plot_index,
            filename=args.output
            )
        logger.info(f"Saved eigenvector to {args.output}")
    
    # Create and print AE3DEigenvector
    try:
        eigvec = solver.create_ae3d_eigenvector(vector_index=args.plot_index)
        logger.info(f"Created AE3DEigenvector"
                     "with eigenvalue {eigvec.eigenvalue}")
        logger.info(f"Contains {len(eigvec.harmonics)} harmonics "
                    f"with {len(eigvec.s_coords)} radial points")
    except Exception as e:
        logger.error(f"Failed to create AE3DEigenvector: {str(e)}")
    
    # Plot if requested
    if args.plot:
        solver.plot_eigenvalues(
            save_path=os.path.join(args.save_plots, 'eigenvalues.png') if args.save_plots else None,
            compare_with_ae3d=args.compare
        )
        solver.plot_eigenvector(
            vector_index=args.plot_index, 
            save_path=os.path.join(args.save_plots, f'eigenvector_{args.plot_index}.png') if args.save_plots else None
        )
