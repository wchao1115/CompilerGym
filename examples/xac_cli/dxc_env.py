import os
from typing import Iterable, List, Dict, DefaultDict
from compiler_gym.util.registration import register
from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import File
from compiler_gym.spaces import Reward
from compiler_gym.datasets import Benchmark, BenchmarkSource, Dataset
from compiler_gym.datasets.uri import BenchmarkUri

class OccupancyReward(Reward):
    """A reward based on changes of theoretical shader occupancy as determined by compiled ISA instructions."""

    def __init__(self):
        super().__init__(
            name="occupancy",
            observation_spaces=["isa"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self._last_occupancy = None

    def reset(self, benchmark: str, observation_view):
        del benchmark
        self._last_occupancy = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self._last_occupancy is None:
            self._last_occupancy = observations[0].occupancy

        reward = float(self._last_occupancy - observations[0].occupancy)
        self._last_occupancy = observations[0].occupancy
        return reward

class ShaderSourceDataset(Dataset):
    """A dataset comprising a collection of HLSL shader programs."""

    def __init__(self, name:str, path:str, *args, **kwargs):
        source_ext = ".hlsl"
        scheme = "benchmark"
        base_uri = scheme + "://" + name
        super().__init__(
            name=name,
            license="Microsoft Internal",
            description=self.__class__.__doc__,
            )

        def _content_from_path(path:str) -> bytes:
            with open(path, 'rb') as file:
                content = file.read()
            return content

        prereqs:List[BenchmarkSource] = []
        for file in [_ for _ in os.listdir(path) if _.lower().endswith(source_ext)]:
            prereqs.append(BenchmarkSource(filename=file, contents=_content_from_path(path=os.path.join(path, file))))

        self._benchmarks:DefaultDict[Benchmark] = {}
        for file in [_ for _ in os.listdir(os.path.join(path, scheme)) if _.lower().endswith(source_ext)]:
            # Each benchmark has all the source files needed and a file name that identifies which one has the program entry point.
            uri_path = '/' + os.path.splitext(file)[0].lower()
            benchmark = Benchmark(
                proto=BenchmarkProto(uri=base_uri + uri_path, program=File(uri=file)),
                sources=prereqs
                )
            benchmark.add_source(BenchmarkSource(filename=file, contents=_content_from_path(path=os.path.join(path, scheme, file))))
            self._benchmarks[uri_path] = benchmark

    def benchmark_uris(self) -> Iterable[str]:
        yield from [str(detail.uri) for _, detail in self._benchmarks.items()]

    def benchmark_from_parsed_uri(self, uri: BenchmarkUri) -> Benchmark:
        if uri.path in self._benchmarks:
            return self._benchmarks[uri.path]
        else:
            raise LookupError("Unknown benchmark program name")

def register_dxc_env(port=50051):
    """
    Register a client service environment for DXC compilation. DXC is Windows-only so it'll run
    within a Windows host service that can be accessed from a Linux client via gRPC.
    """
    register(
        id="dxc-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": "host.docker.internal:" + str(port),
            "rewards": [OccupancyReward()],
            "datasets": [
                ShaderSourceDataset(name="dxc-v0-gemm", path=os.path.join(os.getcwd(), "examples/xac_cli/dataset/gemm"))
            ]
        },
    )
