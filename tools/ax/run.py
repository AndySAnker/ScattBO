from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from ax.service.managed_loop import optimize

from ScattBO.parameters.benchmark_parameters import BenchmarkParameters
from ScattBO.benchmark.benchmark import Benchmark

# One example:

benchmark = Benchmark("small", "Gr", "simulated")


def benchmark_wrapper_for_ax(parameters):
    params = BenchmarkParameters(**parameters)
    val = benchmark(params)

    if np.isnan(val):
        return {"value": (-20.0, 0.0)}
    else:
        return {"value": (val, 0.0)}


best_parameters, best_values, experiment, model = optimize(
    parameters=[
        {
            "name": "pH",
            "type": "range",
            "bounds": list(benchmark.search_space["pH"]),
        },
        {
            "name": "pressure",
            "type": "range",
            "bounds": list(benchmark.search_space["pressure"]),
        },
        {
            "name": "solvent",
            "type": "choice",
            "values": ["Ethanol", "Methanol"],
        },
    ],
    evaluation_function=benchmark_wrapper_for_ax,
    objective_name="value",
    minimize=False,
)

print(best_values)
print(best_parameters)
