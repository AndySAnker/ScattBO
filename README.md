# ScattBO Benchmark - Bayesian optimisation for materials discovery

A self-driving laboratory (SDL) is an autonomous platform that conducts machine learning (ML) selected experiments to achieve a user-defined objective. An objective can be to synthesise a specific material.[1]
Such an SDL will synthesise a material, evaluate if this is the target material and if necessary optimise the synthesis parameters for the next synthesis. One way to evaluate if the material is the target material is by measuring scattering data and comparing that to the scattering pattern of the target material.

Here, we present a Python-based benchmark (ScattBO) that simulates a SDL for materials discovery. Based on a set of synthesis parameters, the function "synthesis" a structure, calculate the scattering pattern[2] and compare this to the scattering pattern of the target structure. Note: Scattering data may not be enough to definitely 
validate that the target material has been synthesised.[3] The benchmark can include other types of data as long they can be simulated.

**Synthesis parameters**:
  - pH (float):       The pH value, which scales the size of the structure. Range: [0, 14]
  - pressure (float): The pressure value, which controls the lattice constant of the structure. Range: [0, 100]
  - solvent (int):    The solvent type, which determines the structure type for large clusters. 
                      0 for 'Ethanol', 1 for 'Methanol', 2 for 'Water', 3 for 'Others'

**Features**:
  - Continous parameters (pH and pressure) and discrete parameter (solvent).
  - Two sizes of benchmark is provided: 1) ScatterBO_small_benchmark and 2) ScatterBO_large_benchmark.
  - Possibility of two different objectives: 1) The scattering data in Q-space (Sq) or 2) the scattering data in r-space (Gr).
  - Possibility of multi-objective optimisation using both Sq and Gr.
  - Possibility of four target scattering patterns: 1) simulated Sq, 2) simulated Gr, 3) experimental Sq, and 4) experimental Gr.
  - OBS: The scattering pattern calculations can be slow on CPU. **It is recommended to use GPU.**

# Scoreboard

<p align="center"><i>These scoreboards represent the performance of different BO algorithms on various of the ScattBO benchmarks. If you have new scores to report, feel free to contact us.</i></p>

<table>
<tr>
<td>

#### Scoreboard for Sq - Simulated
| Team | Steps for Convergence<sup>1</sup> | Name of Algorithm |
|:----:|:---------------------------------:|:-----------------:|
| #24 | 4808                              | Bruteforce in structure space |

</td>
<td>

#### Scoreboard for Sq - Experimental
| Team | Steps for Convergence<sup>3</sup> | Name of Algorithm |
|:----:|:---------------------------------:|:-----------------:|
| #24 | 4808                              | Bruteforce in structure space |

</td>
</tr>
<tr>
<td>

#### Scoreboard for Gr - Simulated
| Team | Steps for Convergence<sup>2</sup> | Name of Algorithm |
|:----:|:---------------------------------:|:-----------------:|
| #24 | 4808                              | Bruteforce in structure space |

</td>
<td>

#### Scoreboard for Gr - Experimental
| Team | Steps for Convergence<sup>4</sup> | Name of Algorithm |
|:----:|:---------------------------------:|:-----------------:|
| #24 | 4808                              | Bruteforce in structure space |

</td>
</tr>
<tr>
<td>

#### Scoreboard for Multi-objective - Simulated
| Team | Steps for Convergence<sup>5</sup> | Name of Algorithm |
|:----:|:---------------------------------:|:-----------------:|
| #24 | 4808                              | Bruteforce in structure space |

</td>
<td>

#### Scoreboard for Multi-objective - Experimental
| Team | Steps for Convergence<sup>5</sup> | Name of Algorithm |
|:----:|:---------------------------------:|:-----------------:|
| #24 | 4808                              | Bruteforce in structure space |

</td>
</tr>
</table>

<sup>1</sup> Steps for Convergence for simulated Sq data is defined as the number of steps until Rwp < 0.04.<br>
<sup>2</sup> Steps for Convergence for simulated Gr data is defined as the number of steps until Rwp < 0.04.<br>
<sup>3</sup> Steps for Convergence for experimental Sq data is defined as the number of steps until Rwp < 0.79.<br>
<sup>4</sup> Steps for Convergence for experimental Gr data is defined as the number of steps until Rwp < 0.84.<br>
<sup>5</sup> Steps for Convergence for multi-objective optimisation is defined for both above criteria.

# Usage
See (https://github.com/AndySAnker/ScattBO/blob/main/ScatBO_SingleObjective.ipynb) for examples of single-objective optimisation with [Dragonfly](https://github.com/dragonfly/dragonfly/tree/master).

## Example usage with [Dragonfly](https://github.com/dragonfly/dragonfly/tree/master)
```python
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
from utils.ScattBO import generate_structure, ScatterBO_small_benchmark, ScatterBO_large_benchmark

from dragonfly import minimise_function

def benchmark_wrapper(params):
    """
    Wrapper function for the ScatterBO_small_benchmark function.

    Parameters:
    params (list): The parameters to pass to the ScatterBO_small_benchmark function.

    simulated_or_experimental (str): If 'simulated', use the filename 'Data/Gr/Target_PDF_small_benchmark.npy'. 
                                     If 'experimental', use the filename 'T2_0p7boro_15hrs_powder.npy'. Default is 'simulated'.
    
    scatteringfunction= (str): The scattering function to use. 'Gr' for pair distribution function, 'Sq' for structure factor. Default is 'Sq'.

    Returns:
    The result of the ScatterBO_small_benchmark function.
    """
    return ScatterBO_small_benchmark(params, plot=False, simulated_or_experimental='simulated', scatteringfunction='Gr')

# Define the domain for each parameter
domain = [
    [2, 12],  # pH values range from 2 to 12
    [20, 80],  # pressure values range from 20 to 80
    [0, 1]  # solvent can be 0 ('Ethanol') or 1 ('Methanol')
]

# Define the maximum capital
max_capital = 10

# Minimize the function
min_val, min_pt, history = minimise_function(benchmark_wrapper, domain, max_capital)

```

## Visualise results
```python
# Print the minimum value and the corresponding parameters
print ("min_val: ", min_val)
print ("min_pt: ", min_pt)

# Use the parameters that minimize the function as input to the generate_structure function
pH, pressure, solvent = min_pt
cluster = generate_structure(pH, pressure, solvent, atom='Au')
print ("Cluster type: ", cluster.structure_type)
print ("Number of atoms: ", len(cluster))

# Use the parameters that minimize the function as input to the ScatterBO_small_benchmark function
ScatterBO_small_benchmark((pH, pressure, solvent), plot=True, simulated_or_experimental='simulated', scatteringfunction='Gr')

```

# Contributing to the software

We welcome contributions to our software! To contribute, please follow these steps:

1. Fork the repository.
2. Make your changes in a new branch.
3. Submit a pull request.

We'll review your changes and merge them if they are appropriate. 

## Reporting issues

If you encounter any issues or problems with our software, please report them by opening an issue on our GitHub repository. Please include as much detail as possible, including steps to reproduce the issue and any error messages you received.

## Seeking support

If you need help using our software, please reach out to us on our GitHub repository. We'll do our best to assist you and answer any questions you have.

# References**:

[1] Szymanski, Nathan J., et al., An autonomous laboratory for the accelerated synthesis of novel materials, Nature, 624(7990), 86-91 (2023)

[2] Frederik L. Johansen & Andy S. Anker, et al., A GPU-Accelerated Open-Source Python Package for Calculating Powder Diffraction, Small-Angle-, and Total Scattering with the Debye Scattering Equation, Journal of Open Source Software, 9(94), 6024 (2024)

[3] Leeman, Josh, et al., Challenges in High-Throughput Inorganic Materials Prediction and Autonomous Synthesis, PRX Energy, 3(1), 011002 (2024)
