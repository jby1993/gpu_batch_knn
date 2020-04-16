from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(name='batch_knn',
      ext_modules=[CUDAExtension('batch_knn', ['BatchKnnGpu.cpp','CudaKernels.cu'])],
      cmdclass={'build_ext': BuildExtension})