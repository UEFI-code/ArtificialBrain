from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mySuperExtension',
    ext_modules=[
        CUDAExtension('myBionicCell_cuda', [
            'myBionicCell.cpp',
            'myBionicCellGPU.cu',
            'myBionicCellCPU.cpp'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
