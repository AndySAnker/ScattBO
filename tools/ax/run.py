from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from ax.service.managed_loop import optimize

from benchmark import Benchmark

# One example:

benchmark = Benchmark("large", "Gr", "simulated")


def benchmark_wrapper_for_ax(parameters):
    val = benchmark(
        [
            parameters.get("pH"),
            parameters.get("pressure"),
            parameters.get("solvent"),
        ]
    )

    print(val)
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
            "type": "range",
            "bounds": list(benchmark.search_space["solvent"]),
        },
    ],
    evaluation_function=benchmark_wrapper_for_ax,
    objective_name="value",
    minimize=False,
)

print(best_values)
print(best_parameters)
