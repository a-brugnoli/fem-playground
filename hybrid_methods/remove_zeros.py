import numpy as np

def remove_near_zero_rows_cols(matrix, threshold=1e-10):
    # Remove rows close to zero
    rows_mask = ~np.all(np.abs(matrix) < threshold, axis=1)
    print(rows_mask)
    matrix = matrix[rows_mask]
    
    # Remove columns close to zero
    cols_mask = ~np.all(np.abs(matrix) < threshold, axis=0)
    print(cols_mask)

    matrix = matrix[:, cols_mask]
    
    return matrix

# Example
arr = np.array([
    [0, 0, 3, 0],
    [0, 0, 0, 0],
    [1, 2, 0, 0]
])

result = remove_near_zero_rows_cols(arr, threshold=1e-11)
print(result)