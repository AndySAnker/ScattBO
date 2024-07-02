"""
In this example, we optimize by SOBOL sampling in the search space.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import botorch
import pandas as pd

from benchmark import Benchmark

fn = Benchmark("small", "Gr", "simulated")

max_iter = 50

sobol_samples = torch.quasirandom.SobolEngine(dimension=3, seed=1234).draw(max_iter)

# Transform them to the appropriate domain
sobol_samples[:, 0] = (
    sobol_samples[:, 0] * (fn.search_space["pH"][1] - fn.search_space["pH"][0])
    + fn.search_space["pH"][0]
)
sobol_samples[:, 1] = (
    sobol_samples[:, 1]
    * (fn.search_space["pressure"][1] - fn.search_space["pressure"][0])
    + fn.search_space["pressure"][0]
)
sobol_samples[:, 2] = (
    sobol_samples[:, 2]
    * (fn.search_space["solvent"][1] - fn.search_space["solvent"][0])
    + fn.search_space["solvent"][0]
)

rows = []
for i in range(max_iter):
    print("Evaluating sample", i)
    print(sobol_samples[i])
    val = fn(sobol_samples[i].tolist())
    print(val)
    # Do something with the value
    # For example, save it to a file

    row = {
        "pH": sobol_samples[i][0].item(),
        "pressure": sobol_samples[i][1].item(),
        "solvent": sobol_samples[i][2].item(),
        "value": val,
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("sobol_samples.csv")
