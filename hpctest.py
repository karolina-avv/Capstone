import numpy as np
from multiprocessing import Pool
import os

# Check basic setup
def check_environment():
    print("Running on node:", os.uname().nodename)
    print("CPU count:", os.cpu_count())

# Simple function for parallel processing
def square_number(x):
    return x * x

# Run some parallel computations
def run_parallel_computation():
    with Pool(4) as p:
        results = p.map(square_number, range(10))
    print("Parallel computation results:", results)

# Main function
if __name__ == "__main__":
    check_environment()
    
    # Test numpy array operations
    array = np.array([1, 2, 3, 4, 5])
    print("Numpy array:", array)
    print("Numpy array squared:", np.square(array))
    
    # Run parallel computation
    run_parallel_computation()