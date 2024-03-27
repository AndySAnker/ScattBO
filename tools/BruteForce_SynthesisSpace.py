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

def generate_structure(pH, pressure, solvent, atom='Au'):
    """
    Generate a structure based on the given parameters.

    Parameters:
    pH (float): The pH value, which scales the size of the structure. Range: [0, 14]
    pressure (float): The pressure value, which controls the lattice constant. Range: [0, 100]
    solvent (str): The solvent type, which determines the structure type for small clusters. 
                   'Ethanol', 'Methanol', 'Water', or any other solvent
    atom (str): The atom type. Default is 'Au'.

    Returns:
    cluster: The generated structure.
    """
    # Scale the size of the structure based on pH
    scale_factor = pH / 14  # Normalize pH to range [0, 1]
    noshells = int(scale_factor * 4) + 2  # Scale noshells from 1 to 4
    p = q = r = noshells  # Set p, q, r to noshells
    layers = [noshells] * 3  # Set layers to [noshells, noshells, noshells]
    surfaces=[[1,0,0], [1,1,0], [1,1,1]]  # Set surfaces to [100], [110], [111]

    # Control lattice constant by pressure
    lc = 2 * (pressure / 100) + 2.5  # Scale lattice constant from 2.5 to 4.5 based on pressure

    # Determine the structure type based on the number of atoms and solvent
    num_atoms = noshells ** 3  # Assume the number of atoms is proportional to noshells^3
    if num_atoms > 2000: # approximate 3 nm in diameter
        if solvent == 'Ethanol':
            cluster = FaceCenteredCubic(atom, directions=surfaces, size=layers, latticeconstant=2*np.sqrt(0.5*lc**2))
            cluster.structure_type = 'FaceCenteredCubic'
        elif solvent == 'Methanol':
            cluster = SimpleCubic(atom, directions=surfaces, size=layers, latticeconstant=lc)
            cluster.structure_type = 'SimpleCubic'
        elif solvent == 'Water':
            cluster = BodyCenteredCubic(atom, directions=surfaces, size=layers, latticeconstant=lc)
            cluster.structure_type = 'BodyCenteredCubic'
        else:
            cluster = HexagonalClosedPacked(atom, latticeconstant=(lc, lc*1.633), size=(noshells, noshells, noshells))
            cluster.structure_type = 'HexagonalClosedPacked'
    elif solvent == 'Ethanol':
        cluster = Icosahedron(atom, noshells, 2*np.sqrt(0.5*lc**2))
        cluster.structure_type = 'Icosahedron'
    elif solvent == 'Methanol':
        cluster = Decahedron(atom, p, q, r, 2*np.sqrt(0.5*lc**2))
        cluster.structure_type = 'Decahedron'
    elif solvent == 'Water':
        cluster = BodyCenteredCubic(atom, directions=surfaces, size=layers, latticeconstant=lc)
        cluster.structure_type = 'BodyCenteredCubic'
    else:
        cluster = Octahedron(atom, length=noshells, latticeconstant=2*np.sqrt(0.5*lc**2))
        cluster.structure_type = 'Octahedron'

    return cluster

def BruteForce(atom='Au', simulated_or_experimental='simulated', scatteringfunction="Gr"):
    """
    Generate a structure based on the given parameters and calculate the Rwp value for each structure.
    The function returns a list of dictionaries, each containing the pH, pressure, solvent, structure type, number of atoms, and Rwp value for each structure.

    Parameters:
    atom (str): The atom type. Default is 'Au'.
    scatteringfunction (str): The scattering function. Default is 'Gr'.
    simulated_or_experimental (str): The type of data to load. Default is 'simulated'.

    Returns:
    result (list): A list of dictionaries containing information about the generated structures.
    """

    # Load the target PDF data
    x_target, Int_target = LoadData(simulated_or_experimental=simulated_or_experimental, scatteringfunction=scatteringfunction)

    # Set the range of pH, pressure and solvent
    pH_values = np.arange(0, 14, 0.01)
    pressure_values = np.arange(0, 100, 0.5)
    solvent_values = np.arange(0, 4, 1)

    result = []  # Change this to a list
    
    # Create a progress bar
    pbar = tqdm(total=len(pH_values)*len(pressure_values)*len(solvent_values))

    for pH in pH_values:
        for pressure in pressure_values:
            for solvent in solvent_values:

                # Generate the structure
                cluster = generate_structure(pH, pressure, solvent, atom='Au')

                # Check if a structure with the same type and number of atoms already exists
                if any(d['structure_type'] == cluster.structure_type and d['number_of_atoms'] == len(cluster) for d in result):
                    continue

                # Calculate the scattering pattern of the generated structure
                x_sim, Int_sim = calculate_scattering(cluster, scatteringfunction)

                # Interpolate the simulated intensity to the r/q values of the target scattering pattern
                Int_sim_interp = np.interp(x_target, x_sim, Int_sim)

                # Calculate the difference between the simulated and target scattering patterns
                diff = Int_target - Int_sim_interp

                # Calculate the Rwp value
                rwp = np.sqrt(np.sum(diff**2) / np.sum(Int_target**2))

                result.append({  # Append the dictionary to the list
                'pH': pH,
                'pressure': pressure,
                'solvent': solvent,
                'structure_type': cluster.structure_type,  # Add structure type to the dictionary
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
save_dict(result, 'result_simulated_Gr_SynthesisSpace.json')

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
save_dict(result, 'result_simulated_Sq_SynthesisSpace.json')

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
save_dict(result, 'result_experimental_Gr_SynthesisSpace.json')

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
save_dict(result, 'result_experimental_Sq_SynthesisSpace.json')
