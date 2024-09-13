from pathlib import Path

from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic, BodyCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.decahedron import Decahedron
from ase.cluster.octahedron import Octahedron
from debyecalculator import DebyeCalculator
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from ScattBO.parameters.benchmark_parameters import (
    BenchmarkParameters,
    SmallBenchmarkParameters,
    LargeBenchmarkParameters,
    RoboticBenchmarkParameters,
)

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


def calculate_scattering(
    cluster,
    function="Gr",
    qmin=2,
    qmax=10.0,
    qstep=0.01,
    qmin_SAXS=0.01,
    qmax_SAXS=3.0,
    qstep_SAXS=0.01,
    rmin=0,
    rmax=30,
    rstep=0.1,
):
    """
    Calculate a Pair Distribution Function (PDF), Structure Factor (Sq), Form Factor (Fq), Intensity (Iq), or Small Angle X-ray Scattering (SAXS) from a given structure.

    Parameters:
    cluster (ase.Atoms): The atomic structure (from ASE Atoms object) to calculate the function from.
    function (str): The function to calculate. 'Gr' for pair distribution function, 'Sq' for structure factor, 'Fq' for form factor, 'Iq' for intensity, 'SAXS' for small angle X-ray scattering. Default is 'Gr'.
    qmin (float): The minimum q value for the scattering pattern calculation. Default is 2.
    qmax (float): The maximum q value for the scattering pattern calculation. Default is 10.0.
    qstep (float): The step size for q values. Default is 0.01.
    qmin_SAXS (float): The minimum q value for SAXS pattern calculations. Default is 0.01.
    qmax_SAXS (float): The maximum q value for SAXS pattern calculations. Default is 3.0.
    qstep_SAXS (float): The step size for q values in SAXS. Default is 0.01.
    rmin (float): The minimum r value for the Gr pattern calculations. Default is 0.
    rmax (float): The maximum r value for the Gr pattern calculations. Default is 30.
    rstep (float): The step size for r values in Gr. Default is 0.1.

    Returns:
    r/q (numpy.ndarray): The r values (for PDF) or q values (for Sq, Fq, Iq, SAXS) from the calculated function.
    G/S/F/I (numpy.ndarray): The G values (for PDF), S values (for Sq), F values (for Fq), I values (for Iq), or SAXS values from the calculated function.

    Raises:
    AssertionError: If the function parameter is not 'Gr', 'Sq', 'Fq', 'Iq', or 'SAXS'.

    Example:
    >>> cluster = Icosahedron('Au', noshells=7)
    >>> r, G = calculate_scattering(cluster, function='Gr')
    >>> q, S = calculate_scattering(cluster, function='Sq')
    >>> q, F = calculate_scattering(cluster, function='Fq')
    >>> q, I = calculate_scattering(cluster, function='Iq')
    >>> q, SAXS = calculate_scattering(cluster, function='SAXS')
    """
    # Check if the function parameter is valid
    assert function in [
        "Iq",
        "Sq",
        "Fq",
        "Gr",
        "SAXS",
    ], "Function must be 'Iq', 'Sq', 'Fq', 'Gr', or 'SAXS'."

    # Initialise calculator object
    calc = DebyeCalculator(
        qmin=qmin, qmax=qmax, qstep=qstep, rmin=rmin, rmax=rmax, rstep=rstep
    )
    r, Q, I, S, F, G = calc._get_all(structure_source=cluster)

    # Calculate scattering patterns
    if function == "Iq":
        I /= I.max()
        return Q, I
    elif function == "Sq":
        S /= S.max()
        return Q, S
    elif function == "Fq":
        F /= F.max()
        return Q, F
    elif function == "Gr":
        G /= G.max()
        return r, G
    elif function == "SAXS":
        calc.update_parameters(
            qmin_SAXS=qmin_SAXS, qmax_SAXS=qmax_SAXS, qstep_SAXS=qstep_SAXS
        )
        Q_sim, I_sim = calc.iq(structure_source=cluster)
        I_sim /= I_sim.max()
        return Q_sim, I_sim


def generate_structure(benchmark_params: BenchmarkParameters, atom="Au"):
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
    scale_factor = benchmark_params.pH / 14  # Normalize pH to range [0, 1]
    noshells = int(scale_factor * 8) + 2  # Scale noshells from 1 to 4
    p = q = r = noshells  # Set p, q, r to noshells
    layers = [noshells] * 3  # Set layers to [noshells, noshells, noshells]
    surfaces = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]  # Set surfaces to [100], [110], [111]

    # Control lattice constant by pressure
    lc = (
        2 * (benchmark_params.pressure / 100) + 2.5
    )  # Scale lattice constant from 2.5 to 4.5 based on pressure

    # Determine the structure type based on the number of atoms and solvent
    num_atoms = noshells**3  # Assume the number of atoms is proportional to noshells^3
    solvent = benchmark_params.solvent
    if num_atoms > 2000:  # approximate 3 nm in diameter
        if solvent == "Ethanol":
            cluster = FaceCenteredCubic(
                atom,
                directions=surfaces,
                size=layers,
                latticeconstant=2 * np.sqrt(0.5 * lc**2),
            )
            cluster.structure_type = "FaceCenteredCubic"
        elif solvent == "Methanol":
            cluster = SimpleCubic(
                atom, directions=surfaces, size=layers, latticeconstant=lc
            )
            cluster.structure_type = "SimpleCubic"
        elif solvent == "Water":
            cluster = BodyCenteredCubic(
                atom, directions=surfaces, size=layers, latticeconstant=lc
            )
            cluster.structure_type = "BodyCenteredCubic"
        else:
            cluster = HexagonalClosedPacked(
                atom,
                latticeconstant=(lc, lc * 1.633),
                size=(noshells, noshells, noshells),
            )
            cluster.structure_type = "HexagonalClosedPacked"
    elif solvent == "Ethanol":
        cluster = Icosahedron(atom, noshells, 2 * np.sqrt(0.5 * lc**2))
        cluster.structure_type = "Icosahedron"
    elif solvent == "Methanol":
        cluster = Decahedron(atom, p, q, r, 2 * np.sqrt(0.5 * lc**2))
        cluster.structure_type = "Decahedron"
    elif solvent == "Water":
        cluster = BodyCenteredCubic(
            atom, directions=surfaces, size=layers, latticeconstant=lc
        )
        cluster.structure_type = "BodyCenteredCubic"
    else:
        cluster = Octahedron(
            atom, length=noshells, latticeconstant=2 * np.sqrt(0.5 * lc**2)
        )
        cluster.structure_type = "Octahedron"

    return cluster


def LoadData(simulated_or_experimental="simulated", scatteringfunction="Gr"):
    """
    Load scattering data from a file.

    Parameters:
    simulated_or_experimental (str): Specifies whether to load simulated or experimental data.
                                     Default is "simulated".
    scatteringfunction (str): Specifies the type of scattering function.
                              Options are "Gr", "Sq", "Iq", "Fq", and "SAXS". Default is "Gr".

    Returns:
    x_target (numpy.ndarray): The x values from the loaded data.
    Int_target (numpy.ndarray): The intensity values from the loaded data.

    Raises:
    ValueError: If an invalid scatteringfunction is specified.
    """
    # Set the filename based on the simulated_or_experimental and scatteringfunction variables
    if scatteringfunction == "Gr":
        if simulated_or_experimental == "simulated":
            filename = ROOT_DIR / "Data" / "Gr" / "Target_PDF_benchmark.npy"
        else:  # simulated_or_experimental == 'experimental'
            filename = ROOT_DIR / "Data" / "Gr" / "Experimental_PDF.gr"
    elif scatteringfunction == "Sq":
        if simulated_or_experimental == "simulated":
            filename = ROOT_DIR / "Data" / "Sq" / "Target_Sq_benchmark.npy"
        else:  # simulated_or_experimental == 'experimental'
            filename = ROOT_DIR / "Data" / "Sq" / "Experimental_Sq.sq"
    elif scatteringfunction == "Iq":
        if simulated_or_experimental == "simulated":
            filename = ROOT_DIR / "Data" / "Iq" / "Target_Iq_benchmark.npy"
    elif scatteringfunction == "Fq":
        if simulated_or_experimental == "simulated":
            filename = ROOT_DIR / "Data" / "Fq" / "Target_Fq_benchmark.npy"
    elif scatteringfunction == "SAXS":
        if simulated_or_experimental == "simulated":
            filename = ROOT_DIR / "Data" / "SAXS" / "Target_SAXS_benchmark.npy"
    else:
        raise ValueError(f"Invalid scatteringfunction: {scatteringfunction}")

    # Load the data from the file
    data = (
        np.loadtxt(filename, skiprows=25)
        if str(filename).endswith(".gr") or str(filename).endswith(".sq")
        else np.load(filename)
    )

    # Extract the x and intensity values from the data
    x_target = data[:, 0]
    Int_target = data[:, 1]

    return x_target, Int_target


def calculate_loss(x_target, x_sim, Int_target, Int_sim, loss_type="rwp"):
    """
    Calculate the loss between the target and simulated scattering patterns.

    Parameters:
    x_target (numpy.ndarray): The x values of the target scattering pattern.
    x_sim (numpy.ndarray): The x values of the simulated scattering pattern.
    Int_target (numpy.ndarray): The intensity values of the target scattering pattern.
    Int_sim (numpy.ndarray): The intensity values of the simulated scattering pattern.
    loss_type (str): The type of loss to calculate. Options are 'rwp' (default), 'mae', 'mse', and 'smooth_l1'.

    Returns:
    loss (float): The calculated loss value.
    Int_sim_interp (torch.Tensor): The simulated intensity values interpolated to the x values of the target scattering pattern.

    Raises:
    ValueError: If an invalid loss_type is specified.
    """
    # Convert numpy arrays to PyTorch tensors
    x_target = torch.tensor(x_target)
    x_sim = torch.tensor(x_sim)
    Int_target = torch.tensor(Int_target)
    Int_sim = torch.tensor(Int_sim)

    # Interpolate the simulated intensity to the r/q values of the target scattering pattern
    Int_sim_interp = np.interp(x_target.numpy(), x_sim.numpy(), Int_sim.numpy())
    Int_sim_interp = torch.tensor(Int_sim_interp)

    # Calculate the difference between the simulated and target scattering patterns
    diff = Int_target - Int_sim_interp

    if loss_type == "rwp":
        # Calculate the goodness of fit / loss value
        loss = torch.sqrt(torch.sum(diff**2) / torch.sum(Int_target**2))
    elif loss_type == "mae":
        loss = nn.L1Loss()(Int_target, Int_sim_interp)
    elif loss_type == "mse":
        loss = nn.MSELoss()(Int_target, Int_sim_interp)
    elif loss_type == "smooth_l1":
        loss = nn.SmoothL1Loss()(Int_target, Int_sim_interp)
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")

    return loss.item(), Int_sim_interp


def ScatterBO_small_benchmark(
    params: SmallBenchmarkParameters,
    plot=False,
    simulated_or_experimental="simulated",
    scatteringfunction="Gr",
    loss_type="rwp",
    qmin=2,
    qmax=10.0,
    qstep=0.01,
    rmin=0,
    rmax=30,
    rstep=0.1,
    qmin_SAXS=0.01,
    qmax_SAXS=3.0,
    qstep_SAXS=0.01,
):
    """
    Simulate a scattering pattern from synthesis parameters, load a target scattering pattern, and calculate the similarity between them.

    Parameters:
    params (SmallBenchmarkParameters): A pydantic model containing the following parameters:
        pH (float): The pH value, which scales the size of the structure. Range: [2, 12]
        pressure (float): The pressure value, which controls the lattice constant. Range: [15, 80]
        solvent (int): The solvent type, which determines the structure type for small clusters.
                        0 for 'Ethanol', 1 for 'Methanol'
    plot (bool): If True, plot the simulated and target scattering patterns. Default is False.
    simulated_or_experimental (str): If 'simulated', use the filename 'Data/Gr/Target_[XXXX]_benchmark.npy'.
                                     If 'experimental', use the filename 'T2_0p7boro_15hrs_powder.npy'. Default is 'simulated'.
    scatteringfunction (str): The scattering function to use. 'Gr' for pair distribution function, 'Sq' for structure factor,
                              'Iq' for intensity vs q, 'Fq' for form factor, and 'SAXS' for small-angle X-ray scattering. Default is 'Gr'.
    loss_type (str): The type of loss to calculate. Options are 'rwp' (default), 'mae', 'mse', and 'smooth_l1'.
    qmin (float): The minimum q value for the scattering pattern calculation. Default is 2.
    qmax (float): The maximum q value for the scattering pattern calculation. Default is 10.0.
    qstep (float): The step size for q values. Default is 0.01.
    qmin_SAXS (float): The minimum q value for SAXS pattern calculations. Default is 0.01.
    qmax_SAXS (float): The maximum q value for SAXS pattern calculations. Default is 3.0.
    qstep_SAXS (float): The step size for q values in SAXS. Default is 0.01.
    rmin (float): The minimum r value for the Gr pattern calculations. Default is 0.
    rmax (float): The maximum r value for the Gr pattern calculations. Default is 30.
    rstep (float): The step size for r values in Gr. Default is 0.1.

    Returns:
    loss (float): The loss value is a measure of the difference between the simulated and target scattering patterns.
    """
    # Simulate a scattering pattern from synthesis parameters
    cluster = generate_structure(params, atom="Au")
    x_sim, Int_sim = calculate_scattering(
        cluster,
        function=scatteringfunction,
        qmin=qmin,
        qmax=qmax,
        qstep=qstep,
        rmin=rmin,
        rmax=rmax,
        rstep=rstep,
        qmin_SAXS=qmin_SAXS,
        qmax_SAXS=qmax_SAXS,
        qstep_SAXS=qstep_SAXS,
    )

    # Load the target scattering data
    x_target, Int_target = LoadData(simulated_or_experimental, scatteringfunction)

    # Calculate the difference between the simulated and target scattering patterns
    loss, Int_sim_interp = calculate_loss(
        x_target, x_sim, Int_target, Int_sim, loss_type
    )

    # If plot is True, generate an interactive plot of the target and simulated scattering patterns
    if plot:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_target, y=Int_target, mode="lines", name="Target scattering pattern"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_target,
                y=Int_sim_interp,
                mode="lines",
                name="Simulated scattering pattern",
            )
        )
        fig.show()

    return loss


def ScatterBO_large_benchmark(
    params: LargeBenchmarkParameters,
    plot=False,
    simulated_or_experimental="simulated",
    scatteringfunction="Gr",
    loss_type="rwp",
    qmin=2,
    qmax=10.0,
    qstep=0.01,
    rmin=0,
    rmax=30,
    rstep=0.1,
    qmin_SAXS=0.01,
    qmax_SAXS=3.0,
    qstep_SAXS=0.01,
):
    """
    Simulate a scattering pattern from synthesis parameters, load a target scattering pattern, and calculate the similarity between them.

    Parameters:
    params (tuple): A tuple containing the following parameters:
        pH (float): The pH value, which scales the size of the structure. Range: [0, 14]
        pressure (float): The pressure value, which controls the lattice constant. Range: [0, 100]
        solvent (int): The solvent type, which determines the structure type for large clusters.
                        0 for 'Ethanol', 1 for 'Methanol', 2 for 'Water', 3 for 'Others'
    plot (bool): If True, plot the simulated and target PDFs. Default is False.
    simulated_or_experimental (str): If 'simulated', use the filename 'Data/Gr/Target_[XXXX]_benchmark.npy'.
                                     If 'experimental', use the filename 'T2_0p7boro_15hrs_powder.npy'. Default is 'simulated'.
    scatteringfunction (str): The scattering function to use. 'Gr' for pair distribution function, 'Sq' for structure factor,
                              'Iq' for intensity vs q, 'Fq' for form factor, and 'SAXS' for small-angle X-ray scattering. Default is 'Gr'.
    loss_type (str): The type of loss to calculate. Options are 'rwp' (default), 'mae', 'mse', and 'smooth_l1'.
    qmin (float): The minimum q value for the scattering pattern calculation. Default is 2.
    qmax (float): The maximum q value for the scattering pattern calculation. Default is 10.0.
    qstep (float): The step size for q values. Default is 0.01.
    qmin_SAXS (float): The minimum q value for SAXS pattern calculations. Default is 0.01.
    qmax_SAXS (float): The maximum q value for SAXS pattern calculations. Default is 3.0.
    qstep_SAXS (float): The step size for q values in SAXS. Default is 0.01.
    rmin (float): The minimum r value for the Gr pattern calculations. Default is 0.
    rmax (float): The maximum r value for the Gr pattern calculations. Default is 30.
    rstep (float): The step size for r values in Gr. Default is 0.1.

    Returns:
    loss (float): The loss value is a measure of the difference between the simulated and target scattering patterns.
    """
    # Simulate a scattering pattern from synthesis parameters
    cluster = generate_structure(params, atom="Au")
    x_sim, Int_sim = calculate_scattering(
        cluster,
        function=scatteringfunction,
        qmin=qmin,
        qmax=qmax,
        qstep=qstep,
        rmin=rmin,
        rmax=rmax,
        rstep=rstep,
        qmin_SAXS=qmin_SAXS,
        qmax_SAXS=qmax_SAXS,
        qstep_SAXS=qstep_SAXS,
    )

    # Load the target scattering data
    x_target, Int_target = LoadData(simulated_or_experimental, scatteringfunction)

    # Calculate the difference between the simulated and target scattering patterns
    loss, Int_sim_interp = calculate_loss(
        x_target, x_sim, Int_target, Int_sim, loss_type
    )

    # If plot is True, generate an interactive plot of the target and simulated scattering patterns
    if plot:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_target, y=Int_target, mode="lines", name="Target scattering pattern"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_target,
                y=Int_sim_interp,
                mode="lines",
                name="Simulated scattering pattern",
            )
        )
        fig.show()

    return loss


def generate_structure_robotic(params: RoboticBenchmarkParameters, atom: str = "Au"):
    """
    Generate a structure based on the given parameters.

    Parameters:
    temperature (float): The temperature in celsius, which scales the lattice constant. Range: [20, 70]
    uv (int): 15 UV lamps, which controls the size of the structure. Range: [0, 15]
    uvA (int): 7 UV-A lamps, which controls the size of the structure. Range: [0, 7]
    LED (int): 7 LED lamps, which controls the size of the structure. Range: [0, 7]
    pump[A-F]_volume (float): The volume of pump [A-F], which controls the amount of solution in vessel [A-F]. Range: [0, 5]
    pump[A-F]_speed (float): The speed of pump [A-F], which controls the speed the solution in vessel [A-F] is added. Range: [2048, 4096]
    mixing_speed (float): The speed of the mixing process. Range: [2048, 4096]
    atom (str): The atom type. Default is 'Au'.

    Returns:
    cluster: The generated structure.
    """
    pumpA_volume = params.pump_a_volume
    pumpB_volume = params.pump_b_volume
    pumpC_volume = params.pump_c_volume
    pumpD_volume = params.pump_d_volume
    pumpE_volume = params.pump_e_volume
    pumpF_volume = params.pump_f_volume

    pumpD_speed = params.pump_d_speed
    pumpE_speed = params.pump_e_speed

    total_volume = (
        pumpA_volume
        + pumpB_volume
        + pumpC_volume
        + pumpD_volume
        + pumpE_volume
        + pumpF_volume
    )
    if total_volume != 5:
        raise ValueError("The total volume of all pumps must be 5")

    # Scale the size of the structure based on the number of UV lamps, UV-A lamps, and LED lamps
    scale_factor = (3 * params.uv + 2 * params.uvA + params.LED) / (
        3 * 15 + 2 * 7 + 7
    )  # Normalize to range [0, 1]
    noshells = int(scale_factor * 8) + 2  # Scale noshells from 2 to 10
    p = q = r = noshells  # Set p, q, r to noshells
    layers = [noshells] * 3  # Set layers to [noshells, noshells, noshells]
    surfaces = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]  # Set surfaces to [100], [110], [111]

    # Control lattice constant by temperature
    lc = (
        2 * ((params.temperature - 20) / (70 - 20)) + 2.5
    )  # Scale lattice constant from 2.5 to 4.5 based on temperature

    # Determine the structure type based on the number of atoms
    volume_per_atom = lc**3  # Volume occupied by a single atom
    total_volume = (noshells * lc) ** 3  # Total volume of the structure
    num_atoms = total_volume / volume_per_atom  # Number of atoms

    if num_atoms > 2000:  # approximate 3 nm in diameter
        if pumpA_volume < 0.5:
            cluster = FaceCenteredCubic(
                atom,
                directions=surfaces,
                size=layers,
                latticeconstant=2 * np.sqrt(0.5 * lc**2),
            )
            cluster.structure_type = "FaceCenteredCubic"
        elif pumpB_volume > pumpA_volume:
            cluster = SimpleCubic(
                atom, directions=surfaces, size=layers, latticeconstant=lc
            )
            cluster.structure_type = "SimpleCubic"
        elif pumpC_volume > pumpB_volume:
            cluster = BodyCenteredCubic(
                atom, directions=surfaces, size=layers, latticeconstant=lc
            )
            cluster.structure_type = "BodyCenteredCubic"
        else:
            cluster = HexagonalClosedPacked(
                atom,
                latticeconstant=(lc, lc * 1.633),
                size=(noshells, noshells, noshells),
            )
            cluster.structure_type = "HexagonalClosedPacked"
    else:
        if pumpD_speed > 3500:
            if params.mixing_speed > 3500:
                cluster = Icosahedron(atom, noshells, 2 * np.sqrt(0.5 * lc**2))
                cluster.structure_type = "Icosahedron"
            else:
                cluster = Decahedron(atom, p, q, r, 2 * np.sqrt(0.5 * lc**2))
                cluster.structure_type = "Decahedron"
        elif pumpE_speed < 2500:
            cluster = Decahedron(atom, p, q, r, 2 * np.sqrt(0.5 * lc**2))
            cluster.structure_type = "Decahedron"
        elif pumpF_volume > 1:
            cluster = BodyCenteredCubic(
                atom, directions=surfaces, size=layers, latticeconstant=lc
            )
            cluster.structure_type = "BodyCenteredCubic"
        else:
            cluster = Octahedron(
                atom, length=noshells, latticeconstant=2 * np.sqrt(0.5 * lc**2)
            )
            cluster.structure_type = "Octahedron"

    return cluster


def ScatterBO_robotic_benchmark(
    params: RoboticBenchmarkParameters,
    plot=False,
    simulated_or_experimental="simulated",
    scatteringfunction="Gr",
    loss_type="rwp",
    qmin=2,
    qmax=10.0,
    qstep=0.01,
    rmin=0,
    rmax=30,
    rstep=0.1,
    qmin_SAXS=0.01,
    qmax_SAXS=3.0,
    qstep_SAXS=0.01,
):
    """
    Simulate a scattering pattern from synthesis parameters, load a target scattering pattern, and calculate the similarity between them.

    Parameters:
    params (tuple): A tuple containing the following parameters:
        temperature (float): The temperature in celsius, which scales the lattice constant. Range: [20, 70]
        uv (int): 15 UV lamps, which controls the size of the structure. Range: [0, 15]
        uvA (int): 7 UV-A lamps, which controls the size of the structure. Range: [0, 7]
        LED (int): 7 LED lamps, which controls the size of the structure. Range: [0, 7]
        pump[A-F]_volume (float): The volume of pump [A-F], which controls the amount of solution in vessel [A-F]. Range: [0, 5]
        pump[A-F]_speed (float): The speed of pump [A-F], which controls the speed the solution in vessel [A-F] is added. Range: [2048, 4096]
        mixing_speed (float): The speed of the mixing process. Range: [2048, 4096]
        atom (str): The atom type. Default is 'Au'.
    plot (bool): If True, plot the simulated and target PDFs. Default is False.
    simulated_or_experimental (str): If 'simulated', use the filename 'Data/Gr/Target_[XXXX]_benchmark.npy'.
                                     If 'experimental', use the filename 'T2_0p7boro_15hrs_powder.npy'. Default is 'simulated'.
    scatteringfunction (str): The scattering function to use. 'Gr' for pair distribution function, 'Sq' for structure factor,
                              'Iq' for intensity vs q, 'Fq' for form factor, and 'SAXS' for small-angle X-ray scattering. Default is 'Gr'.
    loss_type (str): The type of loss to calculate. Options are 'rwp' (default), 'mae', 'mse', and 'smooth_l1'.
    qmin (float): The minimum q value for the scattering pattern calculation. Default is 2.
    qmax (float): The maximum q value for the scattering pattern calculation. Default is 10.0.
    qstep (float): The step size for q values. Default is 0.01.
    qmin_SAXS (float): The minimum q value for SAXS pattern calculations. Default is 0.01.
    qmax_SAXS (float): The maximum q value for SAXS pattern calculations. Default is 3.0.
    qstep_SAXS (float): The step size for q values in SAXS. Default is 0.01.
    rmin (float): The minimum r value for the Gr pattern calculations. Default is 0.
    rmax (float): The maximum r value for the Gr pattern calculations. Default is 30.
    rstep (float): The step size for r values in Gr. Default is 0.1.

    Returns:
    loss (float): The loss value is a measure of the difference between the simulated and target scattering patterns.
    """
    # Simulate a scattering pattern from synthesis parameters
    cluster = generate_structure_robotic(params)
    x_sim, Int_sim = calculate_scattering(
        cluster,
        function=scatteringfunction,
        qmin=qmin,
        qmax=qmax,
        qstep=qstep,
        rmin=rmin,
        rmax=rmax,
        rstep=rstep,
        qmin_SAXS=qmin_SAXS,
        qmax_SAXS=qmax_SAXS,
        qstep_SAXS=qstep_SAXS,
    )

    # Load the target scattering data
    x_target, Int_target = LoadData(simulated_or_experimental, scatteringfunction)

    # Calculate the difference between the simulated and target scattering patterns
    loss, Int_sim_interp = calculate_loss(
        x_target, x_sim, Int_target, Int_sim, loss_type
    )

    # If plot is True, generate an interactive plot of the target and simulated scattering patterns
    if plot:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_target, y=Int_target, mode="lines", name="Target scattering pattern"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_target,
                y=Int_sim_interp,
                mode="lines",
                name="Simulated scattering pattern",
            )
        )
        fig.show()

    return loss


if __name__ == "__main__":
    LoadData()
