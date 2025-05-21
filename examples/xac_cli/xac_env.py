import os
import random
from typing import Iterable
from compiler_gym.util.registration import register
from compiler_gym.service.proto import File, Benchmark as BenchmarkProto
from compiler_gym.spaces import Reward
from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri

class OccupancyReward(Reward):
    """A reward based on changes of estimated shader occupancy as determined by the compiled ISA instructions."""

    occupancy_index = 5 # See: /xac/server/dxc_session.py

    def __init__(self):
        super().__init__(
            name="occupancy",
            observation_spaces=["metrics"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )

    def reset(self, benchmark: str, observation_view):
        self._last_occupancy = 0

    def update(self, action, observations, observation_view):
        occupancy = observations[0][self.occupancy_index]        
        reward = float(occupancy - self._last_occupancy)
        self._last_occupancy = occupancy
        return reward
    
    def get_occupancy(self, observations):
        return observations[0][self.occupancy_index]
    
    @property
    def last_occupancy(self):
        return self._last_occupancy

class XacDataset(Dataset):
    """A dataset comprising a collection of HLSL shader programs."""

    def __init__(self, name: str, dataset_path: str, path: str, **kwargs):
        source_ext = ".hlsl"
        scheme = "benchmark://"
        benchmark_path = os.path.join(dataset_path, path)
        
        super().__init__(
            name=name,
            license="Microsoft Internal",
            description=self.__class__.__doc__,
            **kwargs
            )
        
        self._benchmarks = {}
        for file in [_ for _ in os.listdir(benchmark_path) if _.lower().endswith(source_ext)]:
            file_path = os.path.join(path, file)
            benchmark = Benchmark(proto=BenchmarkProto(uri=scheme + file_path))
            self._benchmarks[benchmark.uri.path] = benchmark

    def benchmark_uris(self) -> Iterable[str]:
        yield from [str(detail.uri) for _, detail in self._benchmarks.items()]        
        
    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        if uri.path in self._benchmarks:
            return self._benchmarks[uri.path]
        else:
            raise LookupError("Unknown benchmark program name")

def register_xac_env(port=50051):
    """
    Register a client service environment for DXC compilation. DXC is Windows-only so it'll run
    within a Windows host service that can be accessed from a Linux client via gRPC.
    """
    register(
        id="xac-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": "host.docker.internal:" + str(port),
            "rewards": [OccupancyReward()],
            "datasets": [
                XacDataset(name="xac-v0-gemm", dataset_path="/mnt/c/xac/dataset", path="gemm/generated")
            ],
            "observation_space": "states", 
            "reward_space": "occupancy"
        },
    )
