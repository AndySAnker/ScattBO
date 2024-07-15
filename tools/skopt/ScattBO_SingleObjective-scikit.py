from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic, BodyCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.decahedron import Decahedron
from ase.cluster.octahedron import Octahedron
from debyecalculator import DebyeCalculator
import torch
import pickle
import numpy as np
from ScattBO.utils.ScattBO import generate_structure, ScatterBO_small_benchmark, ScatterBO_large_benchmark
from skopt import gp_minimize
from skopt.plots import plot_convergence
# https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html
# More advanced: https://python.plainenglish.io/master-the-power-of-scikit-optimize-a-step-by-step-guide-4346d3a484ea


def run_optimization(simulated_or_experimental='simulated', scatteringfunction='Gr', benchmark_size='small'):
    """
    Run the optimization process.

    Parameters:
    simulated_or_experimental (str): If 'simulated', use the filename 'Data/Gr/Target_PDF_small_benchmark.npy'. 
                                     If 'experimental', use the filename 'T2_0p7boro_15hrs_powder.npy'. Default is 'simulated'.
    scatteringfunction (str): The scattering function to use. 'Gr' for pair distribution function, 'Sq' for structure factor. Default is 'Gr'.
    benchmark_size (str): The size of the benchmark. 'small' for ScatterBO_small_benchmark, 'large' for ScatterBO_large_benchmark. Default is 'small'.
    """
    def benchmark_wrapper(params):
        """
        Wrapper function for the ScatterBO benchmark function.
        """
        if benchmark_size == 'small':
            if scatteringfunction == 'both':
                return ScatterBO_small_benchmark(params, plot=False, simulated_or_experimental=simulated_or_experimental, scatteringfunction='Gr') + \
                       ScatterBO_small_benchmark(params, plot=False, simulated_or_experimental=simulated_or_experimental, scatteringfunction='Sq')
            else:
                return ScatterBO_small_benchmark(params, plot=False, simulated_or_experimental=simulated_or_experimental, scatteringfunction=scatteringfunction)
        elif benchmark_size == 'large':
            if scatteringfunction == 'both':
                return ScatterBO_large_benchmark(params, plot=False, simulated_or_experimental=simulated_or_experimental, scatteringfunction='Gr') + \
                       ScatterBO_large_benchmark(params, plot=False, simulated_or_experimental=simulated_or_experimental, scatteringfunction='Sq')
            else:
                return ScatterBO_large_benchmark(params, plot=False, simulated_or_experimental=simulated_or_experimental, scatteringfunction=scatteringfunction)
        else:
            raise ValueError("Invalid benchmark_size. Expected 'small' or 'large'.")
        
    # Define the domain for each parameter based on the benchmark size
    if benchmark_size == 'small':
        domain = [
            [2, 12],  # pH values range from 2 to 12
            [15, 80],  # pressure values range from 15 to 80
            [0, 1]  # solvent can be 0 ('Ethanol') or 1 ('Methanol')
        ]
    elif benchmark_size == 'large':
        domain = [
            [0, 14],  # pH values range from 0 to 14
            [0, 100],  # pressure values range from 0 to 100
            [0, 3]  # solvent can be 0 ('Ethanol'), 1 ('Methanol'), 2 ('Water'), or 3 ('Other')
        ]
    else:
        raise ValueError("Invalid benchmark_size. Expected 'small' or 'large'.")


    print ("running BO optimisation")
    # Minimize the function
    res = gp_minimize(benchmark_wrapper,                  # the function to minimize
                  domain,             # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=200,         # the number of evaluations of f
                  n_random_starts=10,  # the number of random initialization points
                  noise=0,            # the noise level (optional)
                  random_state=1234)  # the random seed

    print (res)
    print ("Best parameters: ", (res.x, res.fun))

    plot_convergence(res);
    # Print the minimum value and the corresponding parameters
    min_pt = res.x

    # Use the parameters that minimize the function as input to the generate_structure function
    pH, pressure, solvent = min_pt
    cluster = generate_structure(pH, pressure, solvent, atom='Au')
    print ("Cluster type: ", cluster.structure_type)
    print ("Number of atoms: ", len(cluster))

    return res



# Define the possible values for each parameter
simulated_or_experimental_values = ['simulated', 'experimental']
scatteringfunction_values = ['Gr', 'Sq', 'both']
benchmark_size_values = ['large'] # ['small', 'large']

# Iterate over all possible combinations of parameters
for benchmark_size in benchmark_size_values:
    for simulated_or_experimental in simulated_or_experimental_values:
        for scatteringfunction in scatteringfunction_values:
            print(f"Running optimization with simulated_or_experimental='{simulated_or_experimental}', scatteringfunction='{scatteringfunction}', benchmark_size='{benchmark_size}'")
            res = run_optimization(simulated_or_experimental=simulated_or_experimental, scatteringfunction=scatteringfunction, benchmark_size=benchmark_size)
            
            # Save the result to a file
            filename = f'res_{simulated_or_experimental}_{scatteringfunction}_{benchmark_size}-scikit.pkl'
            with open(filename, 'wb') as f:
                pickle.dump((res.func_vals), f)

            print(f"Result saved to {filename}")
