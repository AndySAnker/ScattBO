from typing import Literal, Callable

from utils.ScattBO import (
    ScatterBO_large_benchmark,
    ScatterBO_small_benchmark,
)


class Benchmark:
    def __init__(
        self,
        size: Literal["small", "large"],
        scattering_function: Literal["Gr", "Sq", "both"],
        simulated_or_experimental: Literal["simulated", "experimental"],
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

    def _construct_benchmark(self) -> Callable[[list[float | int | str]], float]:
        match self.size:
            case "small":
                return self._construct_small_benchmark()
            case "large":
                return self._construct_large_benchmark()
            case _:
                raise NotImplementedError

    def _construct_small_benchmark(self) -> Callable[[list[float | int | str]], float]:
        if self.scattering_function == "both":
            functions_to_evaluate = ["Gr", "Sq"]
        else:
            functions_to_evaluate = [self.scattering_function]

        def benchmark(params: list[float | int | str]) -> float:
            val = 0
            for function in functions_to_evaluate:
                val -= ScatterBO_small_benchmark(
                    params,
                    scatteringfunction=function,
                    **self.kwargs_for_benchmark,
                )

            return val

        return benchmark

    def _construct_large_benchmark(self) -> Callable[[list[float | int | str]], float]:
        if self.scattering_function == "both":
            functions_to_evaluate = ["Gr", "Sq"]
        else:
            functions_to_evaluate = [self.scattering_function]

        def benchmark(params: list[float | int | str]) -> float:
            val = 0
            for function in functions_to_evaluate:
                val -= ScatterBO_large_benchmark(
                    params,
                    scatteringfunction=function,
                    **self.kwargs_for_benchmark,
                )
            return val

        return benchmark

    def __call__(self, params: list[float | int | str]) -> float:
        return self.benchmark(params)


if __name__ == "__main__":
    benchmark = Benchmark("small", "both", "simulated")
