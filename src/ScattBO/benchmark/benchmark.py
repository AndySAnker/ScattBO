from typing import Literal, Callable

from ScattBO.utils.ScattBO import (
    ScatterBO_large_benchmark,
    ScatterBO_small_benchmark,
    ScatterBO_robotic_benchmark,
)
from ScattBO.parameters.benchmark_parameters import (
    BenchmarkParameters,
    RoboticBenchmarkParameters,
)


class Benchmark:
    def __init__(
        self,
        size: Literal["small", "large", "robotic"],
        scattering_function: Literal["Iq", "Sq", "Fq", "Gr", "both"],
        scattering_loss: Literal["rwp", ...],
        target_filename: str,
    ):
        self.size = size
        self.scattering_function = scattering_function
        self.simulated_or_experimental = simulated_or_experimental

        if size == "small":
            self.search_space = {
                "pH": (2.0, 12.0),
                "pressure": (15.0, 80.0),
                "solvent": (0, 1),
            }
        elif size == "large":
            self.search_space = {
                "pH": (0.0, 14.0),
                "pressure": (0.0, 100.0),
                "solvent": (0, 3),
            }

        self.kwargs_for_benchmark = {
            "plot": False,
            "simulated_or_experimental": simulated_or_experimental,
        }
        self.benchmark = self._construct_benchmark()

    def _construct_benchmark(self) -> Callable[[BenchmarkParameters], float]:
        match self.size:
            case "small":
                return self._construct_small_benchmark()
            case "large":
                return self._construct_large_benchmark()
            case "robotic":
                return self._construct_robotic_benchmark()
            case _:
                raise NotImplementedError

    def _construct_small_benchmark(self) -> Callable[[BenchmarkParameters], float]:
        if self.scattering_function == "both":
            functions_to_evaluate = ["Gr", "Fq"]
        else:
            functions_to_evaluate = [self.scattering_function]

        def benchmark(params: BenchmarkParameters) -> float:
            val = 0
            for function in functions_to_evaluate:
                val -= ScatterBO_small_benchmark(
                    params,
                    scatteringfunction=function,
                    **self.kwargs_for_benchmark,
                )

            return val

        return benchmark

    def _construct_large_benchmark(self) -> Callable[[BenchmarkParameters], float]:
        if self.scattering_function == "both":
            functions_to_evaluate = ["Gr", "Fq"]
        else:
            functions_to_evaluate = [self.scattering_function]

        def benchmark(params: BenchmarkParameters) -> float:
            val = 0
            for function in functions_to_evaluate:
                val -= ScatterBO_large_benchmark(
                    params,
                    scatteringfunction=function,
                    **self.kwargs_for_benchmark,
                )
            return val

        return benchmark

    def _construct_robotic_benchmark(
        self,
    ) -> Callable[[RoboticBenchmarkParameters], float]:
        scattering_function = self.scattering_function

        def benchmark(params: RoboticBenchmarkParameters) -> float:
            # TODO:
            # If "both", use the robotic_benchmark to run
            # both Gr and Fq.
            if scattering_function == "both":
                # TODO: How do we deal with normalization?
                gr_value = ScatterBO_robotic_benchmark(
                    params,
                    scatteringfunction="Gr",
                    target_filename="...",
                    **self.kwargs_for_benchmark,
                )
                fq_value = ScatterBO_robotic_benchmark(
                    params,
                    scatteringfunction="Fq",
                    target_filename="...",
                    **self.kwargs_for_benchmark,
                )
                return gr_value + fq_value
            else:
                return ScatterBO_robotic_benchmark(
                    params,
                    scatteringfunction=scattering_function,
                    target_filename="...",
                    **self.kwargs_for_benchmark,
                )

        return benchmark

    def __call__(self, params: BenchmarkParameters) -> float:
        return self.benchmark(params)


if __name__ == "__main__":
    benchmark = Benchmark("small", "both", "simulated")
