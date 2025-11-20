from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

versions = [
    "-gencode",
    "arch=compute_80,code=sm_80",
    "-gencode",
    "arch=compute_89,code=sm_89",
    "-gencode",
    "arch=compute_90,code=sm_90",
] # TODO: assumes installed CUDA toolkit supports sm_80 to sm_90

setup(
    name='hada_core',
    ext_modules=[
        CUDAExtension(
            name="hada_core",
            sources=[
                "hadamard_transform.cpp",
                "hadamard_transform_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    # "-lineinfo",
                    '--ptxas-options=--warn-on-local-memory-usage',
                    '--ptxas-options=--warn-on-spills',
                    "--keep",
                ] + versions
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
