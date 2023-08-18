"""Based on https://github.com/ethanfetaya/NRI (MIT License)."""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


class EconomicSim(object):
    def __init__(
        self,
        num_timesteps = 12*20,
        num_objects = 5,
        num_lags = 2,
    ):
        """
        
        Parameters:
        - num_timesteps: Number of observations in the time series.
        - num_objects: Number of time series.
        - num_lags: Number of lags in the VAR process.

        """
        self.num_objects = num_objects
        self.num_timesteps = num_timesteps
        self.num_lags = num_lags
    
    def generate_stationary_phi(self, sparsity_threshold):
        """
        Generate a stationary phi matrix for VAR process with sparsity.
        
        Parameters:
        - sparsity_threshold: Threshold below which coefficients are set to zero.
        
        Returns:
        - phi: Stationary and sparse coefficient matrix of shape (self.num_objects, self.num_objects * self.num_lags).
        """
        max_attempts = 1000
        attempt = 0
        
        while attempt < max_attempts:
            phi = np.random.randn(self.num_objects, self.num_objects * self.num_lags) * 0.1
            
            # Induce sparsity
            phi[np.abs(phi) < sparsity_threshold] = 0
            
            # Companion form
            companion = np.zeros((self.num_objects * self.num_lags, self.num_objects * self.num_lags))
            companion[:self.num_objects, :] = phi
            companion[self.num_objects:, :-self.num_objects] = np.eye((self.num_lags - 1) * self.num_objects)
            
            eigenvalues = np.linalg.eigvals(companion)
            
            if np.all(np.abs(eigenvalues) < 1):
                # All eigenvalues inside the unit circle implies stationarity
                return phi
            
            attempt += 1
        
        raise ValueError(f"Failed to find stationary phi matrix in {max_attempts} attempts")
    
    def simulate_VAR(self, sparsity_threshold=0.05, seed=None, phi=None, c=None):
        """
        Simulate a VAR process.
        
        Parameters:
        - phi: Coefficient matrix. If None, a stationary matrix is generated.
        - c: Constant vector. If None, a default vector is used.
        
        Returns:
        - data: Simulated data of shape (self.num_timesteps, self.num_objects).
        """
        
        if seed is not None:
            np.random.seed(seed)  # Seed for reproducibility

        if phi is None:
            phi = generate_stationary_phi(self.num_objects, self.num_lags, sparsity_threshold)
        
        if c is None:
            c = np.zeros(self.num_objects)
        
        # Initialize
        data = np.zeros((self.num_timesteps, self.num_objects))
        epsilons = np.random.randn(self.num_timesteps, self.num_objects)
        
        for t in range(self.num_lags, self.num_timesteps):
            for l in range(self.num_lags):
                data[t] += np.dot(phi[:, self.num_objects*l:self.num_objects*(l+1)], data[t-l-1])
            data[t] += c + epsilons[t]
        
        return data, phi