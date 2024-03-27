from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic, BodyCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.decahedron import Decahedron
from ase.cluster.octahedron import Octahedron
from debyecalculator import DebyeCalculator
import torch
import numpy as np
import json
from tqdm import tqdm

def calculate_scattering(cluster, scatteringfunction='Gr'):
    """
    Calculate a Pair Distribution Function (PDF) or Structure Factor (Sq) from a given structure.

    Parameters:
    cluster (ase.Atoms): The atomic structure (from ASE Atoms object) to calculate the PDF or Sq from.
    scatteringfunction (str): The scatteringfunction to calculate. 'Gr' for pair distribution function, 'Sq' for structure factor. Default is 'Gr'.

    Returns:
    r/q (numpy.ndarray): The r values (for PDF) or q values (for Sq) from the calculated function.
    G/S (numpy.ndarray): The G values (for PDF) or S values (for Sq) from the calculated function.

    Raises:
    AssertionError: If the scatteringfunction parameter is not 'Gr' or 'Sq'.

    Example:
    >>> cluster = Icosahedron('Au', noshells=7)
    >>> r, G = calculate_scattering(cluster, scatteringfunction='Gr')
    >>> q, S = calculate_scattering(cluster, scatteringfunction='Sq')
    """
    # Check if the scatteringfunction parameter is valid
    assert scatteringfunction in ['Gr', 'Sq'], "scatteringfunction must be 'Gr' or 'Sq'"

    # Initialise calculator object
    calc = DebyeCalculator(qmin=2, qmax=10.0, rmax=30, qstep=0.01)

    # Extract atomic symbols and positions
    symbols = cluster.get_chemical_symbols()
    positions = cluster.get_positions()

    # Convert positions to a torch tensor
    positions_tensor = torch.tensor(positions)

    # Create a structure tuple
    structure_tuple = (symbols, positions_tensor)

    # Calculate Pair Distribution Function or Structure Factor
    if scatteringfunction == 'Gr':
        r, G = calc.gr(structure_source=structure_tuple)
        G /= G.max()
        return r, G
    else:  # scatteringfunction == 'Sq'
        q, S = calc.sq(structure_source=structure_tuple)
        S /= S.max()
        return q, S

def LoadData(simulated_or_experimental='simulated', scatteringfunction='Gr'):
    # Set the filename based on the simulated_or_experimental and scatteringfunction variables
    if scatteringfunction == 'Gr':
        if simulated_or_experimental == 'simulated':
            filename = 'Data/Gr/Target_PDF_benchmark.npy'
        else:  # simulated_or_experimental == 'experimental'
            filename = 'Data/Gr/Experimental_PDF.gr'
    else:  # scatteringfunction == 'Sq'
        if simulated_or_experimental == 'simulated':
            filename = 'Data/Sq/Target_Sq_benchmark.npy'
        else:  # simulated_or_experimental == 'experimental'
            filename = 'Data/Sq/Experimental_Sq.sq'

    data = np.loadtxt(filename, skiprows=25) if filename.endswith('.gr') or filename.endswith('.sq') else np.load(filename)
    x_target = data[:, 0]
    Int_target = data[:, 1]

    return x_target, Int_target

def BruteForce(atom='Au', simulated_or_experimental='simulated', scatteringfunction="Gr"):
    """
    Generate a structure based on the given parameters and calculate the Rwp value for each structure.
    The function returns a dictionary where the keys are the structure types and the values are dictionaries containing
    the noshell, lattice constant (lc), number of atoms, and Rwp value for each structure.

    Parameters:
    atom (str): The atom type. Default is 'Au'.
    scatteringfunction (str): The scattering function. Default is 'Gr'.
    simulated_or_experimental (str): The type of data to load. Default is 'simulated'.

    Returns:
    result (dict): A dictionary containing information about the generated structure.
    """

    # Load the target PDF data
    x_target, Int_target = LoadData(simulated_or_experimental=simulated_or_experimental, scatteringfunction=scatteringfunction)

    # Set the range of noshells, surfaces, and lattice constant
    noshells = range(2, 10)
    surfaces=[[1,0,0], [1,1,0], [1,1,1]]  # Set surfaces to [100], [110], [111]
    lc_list = np.arange(2.5, 4.5, 0.02)
    structure_types = ['FaceCenteredCubic', 'SimpleCubic', 'BodyCenteredCubic', 'HexagonalClosedPacked', 'Icosahedron', 'Octahedron', 'Decahedron']

    result = {structure_type: [] for structure_type in structure_types}
    
    # Create a progress bar
    pbar = tqdm(total=len(noshells)*len(lc_list)*len(structure_types))

    for noshell in noshells:
        p = q = r = noshell  # Set p, q, r to noshell
        for lc in lc_list:
            layers = [noshell] * 3  # Set layers to [noshell, noshell, noshell]
            for structure_type in structure_types:
                # Check if a structure with the same type and number of atoms already exists
                if any(d['number_of_atoms'] == len(cluster) for d in result[structure_type]):
                    continue
                if structure_type == 'FaceCenteredCubic':
                    cluster = FaceCenteredCubic(atom, directions=surfaces, size=layers, latticeconstant=2*np.sqrt(0.5*lc**2))
                elif structure_type == 'SimpleCubic':
                    cluster = SimpleCubic(atom, directions=surfaces, size=layers, latticeconstant=lc)
                elif structure_type == 'BodyCenteredCubic':
                    cluster = BodyCenteredCubic(atom, directions=surfaces, size=layers, latticeconstant=lc)
                elif structure_type == 'HexagonalClosedPacked':
                    cluster = HexagonalClosedPacked(atom, latticeconstant=(lc, lc*1.633), size=(noshell, noshell, noshell))
                elif structure_type == 'Icosahedron':
                    cluster = Icosahedron(atom, noshell, 2*np.sqrt(0.5*lc**2))
                elif structure_type == 'Octahedron':
                    cluster = Octahedron(atom, length=noshell, latticeconstant=2*np.sqrt(0.5*lc**2))
                elif structure_type == 'Decahedron':
                    cluster = Decahedron(atom, p, q, r, 2*np.sqrt(0.5*lc**2))

                # Calculate the scattering pattern of the generated structure
                x_sim, Int_sim = calculate_scattering(cluster, scatteringfunction)

                # Interpolate the simulated intensity to the r/q values of the target scattering pattern
                Int_sim_interp = np.interp(x_target, x_sim, Int_sim)

                # Calculate the difference between the simulated and target scattering patterns
                diff = Int_target - Int_sim_interp

                # Calculate the Rwp value
                rwp = np.sqrt(np.sum(diff**2) / np.sum(Int_target**2))

                result[structure_type].append({
                'noshell': noshell,
                'lc': lc,
                'number_of_atoms': len(cluster),
                'rwp': rwp
                })

                # Update the progress bar
                pbar.update()

    pbar.close()

    return result

# Define a function to save a dictionary to a file
def save_dict(dict_obj, file_name):
    with open(file_name, 'w') as f:
        json.dump(dict_obj, f, indent=4)

print ("Running BruteForce.py with the following parameters: atom=Au, simulated_or_experimental=simulated, scatteringfunction='Gr' ")
# Run the BruteForce function
result = BruteForce(atom='Au', simulated_or_experimental='simulated', scatteringfunction="Gr")
# Sort the results based on the rwp value
sorted_results = {k: sorted(v, key=lambda x: x['rwp']) for k, v in result.items()}
# Print the 5 best fits
for structure_type, results in sorted_results.items():
    print(f"Structure: {structure_type}")
    for i in range(min(5, len(results))):
        print(f"Details: {results[i]}")
save_dict(result, 'result_simulated_Gr_fast_StructureSpace.json')

print ("Running BruteForce.py with the following parameters: atom='Au', simulated_or_experimental='simulated', scatteringfunction='Sq' ")
# Run the BruteForce function
result = BruteForce(atom='Au', simulated_or_experimental='simulated', scatteringfunction="Sq")
# Sort the results based on the rwp value
sorted_results = {k: sorted(v, key=lambda x: x['rwp']) for k, v in result.items()}
# Print the 5 best fits
for structure_type, results in sorted_results.items():
    print(f"Structure: {structure_type}")
    for i in range(min(5, len(results))):
        print(f"Details: {results[i]}")
save_dict(result, 'result_simulated_Sq_fast_StructureSpace.json')

print ("Running BruteForce.py with the following parameters: atom='Au', simulated_or_experimental='experimental', scatteringfunction='Gr' ")
# Run the BruteForce function
result = BruteForce(atom='Au', simulated_or_experimental='experimental', scatteringfunction="Gr")
# Sort the results based on the rwp value
sorted_results = {k: sorted(v, key=lambda x: x['rwp']) for k, v in result.items()}
# Print the 5 best fits
for structure_type, results in sorted_results.items():
    print(f"Structure: {structure_type}")
    for i in range(min(5, len(results))):
        print(f"Details: {results[i]}")
save_dict(result, 'result_experimental_Gr_fast_StructureSpace.json')

print ("Running BruteForce.py with the following parameters: atom='Au', simulated_or_experimental='experimental', scatteringfunction='Sq' ")
# Run the BruteForce function
result = BruteForce(atom='Au', simulated_or_experimental='experimental', scatteringfunction="Sq")
# Sort the results based on the rwp value
sorted_results = {k: sorted(v, key=lambda x: x['rwp']) for k, v in result.items()}
# Print the 5 best fits
for structure_type, results in sorted_results.items():
    print(f"Structure: {structure_type}")
    for i in range(min(5, len(results))):
        print(f"Details: {results[i]}")
save_dict(result, 'result_experimental_Sq_fast_StructureSpace.json')
