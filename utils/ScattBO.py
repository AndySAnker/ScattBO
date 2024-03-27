from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic, BodyCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.decahedron import Decahedron
from ase.cluster.octahedron import Octahedron
from debyecalculator import DebyeCalculator
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def calculate_scattering(cluster, function='Gr'):
    """
    Calculate a Pair Distribution Function (PDF) or Structure Factor (Sq) from a given structure.

    Parameters:
    cluster (ase.Atoms): The atomic structure (from ASE Atoms object) to calculate the PDF or Sq from.
    function (str): The function to calculate. 'Gr' for pair distribution function, 'Sq' for structure factor. Default is 'Gr'.

    Returns:
    r/q (numpy.ndarray): The r values (for PDF) or q values (for Sq) from the calculated function.
    G/S (numpy.ndarray): The G values (for PDF) or S values (for Sq) from the calculated function.

    Raises:
    AssertionError: If the function parameter is not 'Gr' or 'Sq'.

    Example:
    >>> cluster = Icosahedron('Au', noshells=7)
    >>> r, G = calculate_scattering(cluster, function='Gr')
    >>> q, S = calculate_scattering(cluster, function='Sq')
    """
    # Check if the function parameter is valid
    assert function in ['Gr', 'Sq'], "Function must be 'Gr' or 'Sq'"

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
    if function == 'Gr':
        r, G = calc.gr(structure_source=structure_tuple)
        G /= G.max()
        return r, G
    else:  # function == 'Sq'
        q, S = calc.sq(structure_source=structure_tuple)
        S /= S.max()
        return q, S


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

def ScatterBO_small_benchmark(params, plot=False, simulated_or_experimental='simulated', scatteringfunction='Gr'):
    """
    Simulate a PDF from synthesis parameters, load a target PDF, and calculate the Rwp value.

    Parameters:
    params (tuple): A tuple containing the following parameters:
        pH (float): The pH value, which scales the size of the structure. Range: [2, 12]
        pressure (float): The pressure value, which controls the lattice constant. Range: [20, 80]
        solvent (int): The solvent type, which determines the structure type for small clusters. 
                        0 for 'Ethanol', 1 for 'Methanol'
    plot (bool): If True, plot the simulated and target PDFs. Default is False.
    simulated_or_experimental (str): If 'simulated', use the filename 'Data/Gr/Target_PDF_small_benchmark.npy'. 
                                     If 'experimental', use the filename 'T2_0p7boro_15hrs_powder.npy'. Default is 'simulated'.
    scatteringfunction (str): The scattering function to use. 'Gr' for pair distribution function, 'Sq' for structure factor. Default is 'Gr'.

    Returns:
    rwp (float): The Rwp value, which is a measure of the difference between the simulated and target PDFs.
    """
    pH, pressure, solvent = params
    # Map the numerical solvent variable back to a category
    solvent = ['Ethanol', 'Methanol'][round(solvent)]

    # Check if pH is within the expected range
    if not 2 <= pH <= 12:
        raise ValueError("Invalid pH value. Expected a value between 2 and 12, got {}".format(pH))

    # Check if pressure is within the expected range
    if not 20 <= pressure <= 80:
        raise ValueError("Invalid pressure value. Expected a value between 20 and 80, got {}".format(pressure))

    # Check if solvent is either 0 or 1
    if solvent not in ["Ethanol", "Methanol"]:
        raise ValueError("Invalid solvent value. Expected 0 (for 'Ethanol') or 1 (for 'Methanol'), got {}".format(solvent))

    # Simulate a scattering pattern from synthesis parameters
    cluster = generate_structure(pH, pressure, solvent, atom='Au')
    x_sim, Int_sim = calculate_scattering(cluster, function=scatteringfunction)

    # Load the target scattering data
    x_target, Int_target = LoadData(simulated_or_experimental, scatteringfunction)

    # Interpolate the simulated intensity to the r/q values of the target scattering pattern
    Int_sim_interp = np.interp(x_target, x_sim, Int_sim)

    # Calculate the difference between the simulated and target scattering patterns
    diff = Int_target - Int_sim_interp

    # Calculate the Rwp value
    rwp = np.sqrt(np.sum(diff**2) / np.sum(Int_target**2))

    # If plot is True, generate an interactive plot of the target and simulated scattering patterns
    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_target, y=Int_target, mode='lines', name='Target PDF'))
        fig.add_trace(go.Scatter(x=x_target, y=Int_sim_interp, mode='lines', name='Simulated PDF'))
        fig.show()

    return rwp

def ScatterBO_large_benchmark(params, plot=False, simulated_or_experimental='simulated', scatteringfunction='Gr'):
    """
    Simulate a PDF from synthesis parameters, load a target PDF, and calculate the Rwp value.

    Parameters:
    params (tuple): A tuple containing the following parameters:
        pH (float): The pH value, which scales the size of the structure. Range: [0, 14]
        pressure (float): The pressure value, which controls the lattice constant. Range: [0, 100]
        solvent (int): The solvent type, which determines the structure type for large clusters. 
                        0 for 'Ethanol', 1 for 'Methanol', 2 for 'Water', 3 for 'Others'
    plot (bool): If True, plot the simulated and target PDFs. Default is False.
    simulated_or_experimental (str): If 'simulated', use the filename 'Data/Gr/Target_PDF_large_benchmark.npy'. 
                                     If 'experimental', use the filename 'T2_0p7boro_15hrs_powder.npy'. Default is 'simulated'.
    scatteringfunction (str): The scattering function to use. 'Gr' for pair distribution function, 'Sq' for structure factor. Default is 'Gr'.

    Returns:
    rwp (float): The Rwp value, which is a measure of the difference between the simulated and target PDFs.
    """
    pH, pressure, solvent = params
    # Map the numerical solvent variable back to a category
    solvent = ['Ethanol', 'Methanol', 'Water', 'Others'][round(solvent)]

    # Check if pH is within the expected range
    if not 0 <= pH <= 14:
        raise ValueError("Invalid pH value. Expected a value between 0 and 14, got {}".format(pH))

    # Check if pressure is within the expected range
    if not 0 <= pressure <= 100:
        raise ValueError("Invalid pressure value. Expected a value between 0 and 100, got {}".format(pressure))

    # Check if solvent is one of the valid options
    if solvent not in ["Ethanol", "Methanol", "Water", "Others"]:
        raise ValueError("Invalid solvent value. Expected 0 (for 'Ethanol'), 1 (for 'Methanol'), 2 (for 'Water'), or 3 (for 'Others'), got {}".format(solvent))

    # Simulate a scattering pattern from synthesis parameters
    cluster = generate_structure(pH, pressure, solvent, atom='Au')
    r_sim, G_sim = calculate_scattering(cluster, function=scatteringfunction)

    # Load the target scattering data
    x_target, Int_target = LoadData(simulated_or_experimental, scatteringfunction)

    # Interpolate the simulated scattering pattern to the r/q values of the target scattering pattern
    Int_sim_interp = np.interp(x_target, r_sim, G_sim)

    # Calculate the difference between the simulated and target scattering patterns
    diff = Int_target - Int_sim_interp

    # Calculate the Rwp value
    rwp = np.sqrt(np.sum(diff**2) / np.sum(Int_target**2))

    # If plot is True, generate an interactive plot of the target and simulated scattering patterns
    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_target, y=Int_target, mode='lines', name='Target PDF'))
        fig.add_trace(go.Scatter(x=x_target, y=Int_sim_interp, mode='lines', name='Simulated PDF'))
        fig.show()

    return rwp