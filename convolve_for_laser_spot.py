import numpy as np
from scipy.ndimage import convolve1d

def convolve_columns_with_padding(matrix, kernel):
    # Ensure matrix is a 2D numpy array
    matrix = np.array(matrix)
    n_rows, n_cols = matrix.shape
    
    # Create an empty array to store the convolved columns with the same shape
    result = np.zeros((n_rows, n_cols))
    
    # Perform 1D convolution for each column with zero padding
    for col in range(n_cols):
        # Convolve each column and pad with mode='constant' (zero padding)
        result[:, col] = convolve1d(matrix[:, col], kernel, mode='constant')
        
    return result