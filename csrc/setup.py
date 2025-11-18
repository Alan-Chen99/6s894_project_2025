from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

versions = [
    "-gencode",
    "arch=compute_80,code=sm_80",
    "-gencode",
    "arch=compute_89,code=sm_89",
    "-gencode",
    "arch=compute_90,code=sm_90",
]  # TODO: assumes installed CUDA toolkit supports sm_80 to sm_90

setup(
    name="csrc",
    ext_modules=[
        CUDAExtension(
            name="csrc",
            sources=[
                "hadamard_transform.cpp",
                # "hadamard_transform_cuda.cu",
                "main.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++20",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "-lineinfo",
                    "--ptxas-options=--warn-on-local-memory-usage",
                    "--ptxas-options=--warn-on-spills",
                    # c++20 on torch:
                    # warning #3189-D: "module" is parsed as an identifier rather than a keyword
                    "-diag-suppress=3189",
                    # get ptx files in ./build
                    "--keep",
                ]
                + versions,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
