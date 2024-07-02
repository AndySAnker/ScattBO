import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


import numpy as np

import torch
import gpytorch
from gpytorch.constraints.constraints import GreaterThan
from botorch.models import SingleTaskGP
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import AxClient, ObjectiveProperties

from benchmark import Benchmark

generation_strategy = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=3,
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            model_kwargs={
                "surrogate": Surrogate(
                    botorch_model_class=SingleTaskGP,
                    covar_module_class=gpytorch.kernels.RBFKernel,
                    covar_module_options={
                        "lengthscale_prior": gpytorch.priors.LogNormalPrior(
                            1.4 + np.log(3) / 2, 1.73205
                        ),
                        "lengthscale_constraint": GreaterThan(1e-4),
                    },
                    likelihood_class=gpytorch.likelihoods.GaussianLikelihood,
                    likelihood_options={
                        "noise_prior": gpytorch.priors.LogNormalPrior(-4.0, 1.0),
                        "noise_constraint": GreaterThan(1e-4),
                    },
                ),
                "botorch_acqf_class": qLogNoisyExpectedImprovement,
                "acquisition_options": {
                    "prune_baseline": True,
                },
            },
        ),
    ]
)

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


ax_client = AxClient(generation_strategy=generation_strategy)

ax_client.create_experiment(
    name=f"optimizing_benchmark",
    parameters=[
        {
            "name": "pH",
            "type": "range",
            "bounds": list(benchmark.search_space["pH"]),
            "value_type": "float",
        },
        {
            "name": "pressure",
            "type": "range",
            "bounds": list(benchmark.search_space["pressure"]),
            "value_type": "float",
        },
        {
            "name": "solvent",
            "type": "range",
            "bounds": list(benchmark.search_space["solvent"]),
            "value_type": "float",
        },
    ],
    objectives={"value": ObjectiveProperties(minimize=False)},
)

for i in range(50):
    parameters, trial_index = ax_client.get_next_trial()
    val = benchmark_wrapper_for_ax(parameters)
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data=val,
    )

print(ax_client.get_trials_data_frame())
ax_client.get_trials_data_frame().to_csv("ax_samples.csv")
ax_client.save_to_json_file()
